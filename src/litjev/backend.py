"""Two forwards: shared prefix prefill, then independent cached answer branches."""

import threading
from dataclasses import dataclass

import torch

from litjev.decision import RawFieldScores
from litjev.slots import SLOT_FORMAT, compile_slots


@dataclass(frozen=True)
class ModelSettings:
    model_id: str = "Qwen/Qwen3.8-27B"
    revision: str = "main"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    max_input_tokens: int = 16384


class TransformersScorer:
    def __init__(self, model, tokenizer, max_input_tokens=16384):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.lock = threading.Lock()

    @classmethod
    def load(cls, settings: ModelSettings):
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
        )

        config = AutoConfig.from_pretrained(settings.model_id, revision=settings.revision)
        loader = (
            AutoModelForImageTextToText if config.model_type == "qwen3_5" else AutoModelForCausalLM
        )
        model = loader.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            dtype=getattr(torch, settings.dtype),
            device_map=settings.device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(settings.model_id, revision=settings.revision)
        return cls(model, tokenizer, settings.max_input_tokens)

    def score(self, state, schema):
        with self.lock, torch.inference_mode():
            return self._score(state, schema)

    def _compile(self, state, schema):
        return compile_slots(self.tokenizer, state, schema, self.max_input_tokens)

    def _score(self, state, schema):
        compiled = self._compile(state, schema)
        device = self.model.get_input_embeddings().weight.device
        prefix = compiled.input_ids[0][: compiled.prefix_length]
        prefix_ids = torch.tensor([prefix], device=device)
        base = self.model(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            logits_to_keep=1,
        )
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
                torch.ones((len(schema), len(prefix)), device=device, dtype=torch.long),
                suffix_mask.long(),
            ],
            dim=1,
        )
        positions = torch.arange(len(prefix), len(prefix) + width, device=device)
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions[None, :].expand(len(schema), -1),
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
                len(prefix) + sum(map(len, suffixes)),
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
                    "candidate_token_ids": compiled.candidates[i],
                },
            )
            for i, name in enumerate(schema.names)
        )
