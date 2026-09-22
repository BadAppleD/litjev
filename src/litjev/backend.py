"""Two forwards: shared prefix prefill, then independent cached answer branches."""

import threading
from dataclasses import dataclass

import torch

from litjev.decision import RawFieldScores
from litjev.prompting import build_decision_messages
from litjev.slots import SLOT_FORMAT, compile_prefix_slots, compile_slots
from litjev.vision import VisualState, validate_image


@dataclass(frozen=True)
class ModelSettings:
    model_id: str = "Qwen/Qwen3.8-27B"
    revision: str = "main"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    max_input_tokens: int = 16384


class TransformersScorer:
    def __init__(self, model, tokenizer, max_input_tokens=16384, processor=None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.lock = threading.Lock()
        self.processor = processor

    @classmethod
    def load(cls, settings: ModelSettings, *, eager_kernels: bool = False):
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        config = AutoConfig.from_pretrained(settings.model_id, revision=settings.revision)
        quantization = getattr(config, "quantization_config", {})
        quant_method = (
            quantization.get("quant_method") if isinstance(quantization, dict)
            else getattr(quantization, "quant_method", None)
        )
        if config.model_type == "qwen3_5" and quant_method == "fp8":
            # Dense Qwen has gate_proj, not MoE gate. Transformers prefix-matches
            # stale .mlp.gate exclusions and otherwise drops gate_proj FP8 scales.
            exclusions = (
                quantization if isinstance(quantization, dict) else vars(quantization)
            )
            exclusions["modules_to_not_convert"] = [
                name for name in exclusions.get("modules_to_not_convert") or []
                if not name.endswith(".mlp.gate")
            ]
        loader = (
            AutoModelForImageTextToText if config.model_type == "qwen3_5" else AutoModelForCausalLM
        )
        model = loader.from_pretrained(
            settings.model_id,
            config=config,
            revision=settings.revision,
            dtype=getattr(torch, settings.dtype),
            device_map=settings.device_map,
            allow_all_kernels=True,
        )
        if eager_kernels and quant_method == "fp8":
            from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel
            from transformers.integrations.hub_kernels import allow_all_hub_kernels

            with allow_all_hub_kernels():
                load_finegrained_fp8_kernel()
        tokenizer = AutoTokenizer.from_pretrained(settings.model_id, revision=settings.revision)
        processor = (
            AutoProcessor.from_pretrained(settings.model_id, revision=settings.revision)
            if config.model_type == "qwen3_5"
            else None
        )
        return cls(model, tokenizer, settings.max_input_tokens, processor)

    def score(self, state, schema):
        from transformers.integrations.hub_kernels import allow_all_hub_kernels

        # FP8 hub kernels load lazily after model initialization.
        with self.lock, torch.inference_mode(), allow_all_hub_kernels():
            return self._score(state, schema)

    def _compile(self, state, schema):
        return compile_slots(self.tokenizer, state, schema, self.max_input_tokens)

    def _prepare(self, state, schema):
        device = self.model.get_input_embeddings().weight.device
        if isinstance(state, VisualState):
            if self.processor is None or self.model.config.model_type != "qwen3_5":
                raise ValueError("Image decisions require a Qwen qwen3_5 model and processor")
            validate_image(state.image)
            if state.reference_image is not None:
                validate_image(state.reference_image)
            messages = build_decision_messages(state.text, schema)
            content = [{"type": "text", "text": state.text}]
            if state.reference_image is not None:
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": (
                                "Reference image: identity reference; crop scale is not distance."
                            ),
                        },
                        {"type": "image", "image": state.reference_image.convert("RGB")},
                        {
                            "type": "text",
                            "text": "Current image: current position and composition.",
                        },
                    ]
                )
            content.append({"type": "image", "image": state.image.convert("RGB")})
            messages[-1]["content"] = content
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            if "pixel_values" not in inputs or "mm_token_type_ids" not in inputs:
                raise ValueError("Processor must return pixels and multimodal token types")
            compiled = compile_prefix_slots(
                self.tokenizer,
                schema,
                inputs["input_ids"][0].tolist(),
                max_input_tokens=self.max_input_tokens,
            )
            return compiled, {key: value.to(device) for key, value in inputs.items()}
        compiled = self._compile(state, schema)
        prefix = compiled.input_ids[0][: compiled.prefix_length]
        prefix_ids = torch.tensor([prefix], device=device)
        return compiled, {"input_ids": prefix_ids, "attention_mask": torch.ones_like(prefix_ids)}

    def _score(self, state, schema):
        compiled, inputs = self._prepare(state, schema)
        device = inputs["input_ids"].device
        prefix_length = compiled.prefix_length
        base = self.model(**inputs, use_cache=True, logits_to_keep=1)
        cache = base.past_key_values
        reorder = getattr(cache, "reorder_cache", None)
        if reorder is None:
            raise RuntimeError("Model cache does not support branch replication")
        # Supports Qwen hybrid attention: both KV and convolution/recurrent states.
        reorder(torch.zeros(len(schema), dtype=torch.long, device=device))
        suffixes = compiled.slot_ids
        width = max(map(len, suffixes))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("Tokenizer requires a padding or EOS token")
        ids = torch.tensor([row + [pad] * (width - len(row)) for row in suffixes], device=device)
        lengths = torch.tensor(list(map(len, suffixes)), device=device)
        suffix_mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
        mask = torch.cat(
            [
                torch.ones((len(schema), prefix_length), device=device, dtype=torch.long),
                suffix_mask.long(),
            ],
            dim=1,
        )
        positions = torch.arange(prefix_length, prefix_length + width, device=device)
        positions = positions[None, :].expand(len(schema), -1)
        if isinstance(state, VisualState):
            # Image patches consume sequence slots but have 3-D rotary coordinates.
            # Continue after the full image prefix's M-RoPE extent, not its token count.
            delta = self.model.model.rope_deltas
            if delta is None or delta.shape[0] != 1:
                raise RuntimeError("Missing image-prefix M-RoPE state")
            positions = (positions + delta.to(device))[None, :, :].expand(3, -1, -1)
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
        )
        config = getattr(self.model.config, "text_config", self.model.config)
        return tuple(
            RawFieldScores(
                name,
                output.logits[i, len(suffixes[i]) - 1, compiled.candidates[i]]
                .float()
                .cpu()
                .numpy(),
                prefix_length + sum(map(len, suffixes)),
                {
                    "method": "two_forward_cached_branches",
                    "slot_format": SLOT_FORMAT,
                    "module": "lm_head",
                    "last_decoder_layer_index": config.num_hidden_layers - 1,
                    "decoder_layer_count": config.num_hidden_layers,
                    "batch_index": i,
                    "selected_logit_index": len(suffixes[i]) - 1,
                    "prefix_token_count": compiled.prefix_length,
                    "slot_text": compiled.slot_texts[i],
                    "slot_token_ids": compiled.slot_ids[i],
                    "absolute_position": compiled.positions[i],
                    "candidate_labels": list(schema[name].choices),
                    "candidate_codes": compiled.candidate_codes[i],
                    "candidate_token_ids": compiled.candidates[i],
                    "observation_modality": "image" if isinstance(state, VisualState) else "text",
                    "image_grid_thw": inputs["image_grid_thw"].tolist()
                    if "image_grid_thw" in inputs
                    else None,
                    "readout_rope_positions": positions[:, i, len(suffixes[i]) - 1].tolist()
                    if positions.ndim == 3
                    else None,
                },
            )
            for i, name in enumerate(schema.names)
        )
