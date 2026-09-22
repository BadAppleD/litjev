from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import transformers
from transformers.quantizers.quantizers_utils import should_convert_module

from litjev.backend import ModelSettings, TransformersScorer


def check_load(as_object, model_type, method):
    gates = [f"model.language_model.layers.{i}.mlp.gate" for i in range(64)]
    retained = ["model.visual", "lm_head"]
    quantization = {"quant_method": method, "modules_to_not_convert": gates + retained}
    if as_object:
        quantization = SimpleNamespace(**quantization)
    config = SimpleNamespace(model_type=model_type, quantization_config=quantization)
    loader = Mock(return_value=Mock())
    with ExitStack() as patches:
        patches.enter_context(patch.object(transformers.AutoConfig, "from_pretrained", return_value=config))
        for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
            patches.enter_context(patch.object(getattr(transformers, name), "from_pretrained", loader))
        for name in ("AutoTokenizer", "AutoProcessor"):
            patches.enter_context(patch.object(getattr(transformers, name), "from_pretrained", Mock()))
        TransformersScorer.load(ModelSettings(model_id="unused"))

    assert loader.call_args.kwargs["config"] is config
    exclusions = (vars(quantization) if as_object else quantization)["modules_to_not_convert"]
    if model_type == "qwen3_5" and method == "fp8":
        assert exclusions == retained
        assert all(should_convert_module(gate + "_proj", exclusions) for gate in gates)
        assert not should_convert_module("model.visual.blocks.0.proj", exclusions)
        assert not should_convert_module("lm_head", exclusions)
    else:
        assert exclusions == gates + retained


class FP8LoadingTest(unittest.TestCase):
    def test_dense_gate_proj_scales_and_unchanged_other_configs(self):
        for as_object in (False, True):
            for model_type, method in (("qwen3_5", "fp8"), ("qwen3_5_moe", "fp8"), ("qwen3_5", "other")):
                with self.subTest(as_object=as_object, model_type=model_type, method=method):
                    check_load(as_object, model_type, method)


if __name__ == "__main__":
    unittest.main()
