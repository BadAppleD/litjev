from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys

import torch


EXPECTED = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.14.1",
    "fla-core": "0.5.2",
    "flashinfer-python": "0.6.16.post3",
    "kernels": "0.15.2",
    "kernels-data": "0.16.2",
    "causal-conv1d": "1.7.0+sm120",
}


def main() -> None:
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    for name, expected in EXPECTED.items():
        if not versions[name].startswith(expected):
            raise RuntimeError(f"{name}={versions[name]}, expected {expected}")
    for module in ("causal_conv1d", "fla", "flashinfer"):
        __import__(module)
    if torch.version.cuda != "13.0" or not torch.cuda.is_available():
        raise RuntimeError(f"expected CUDA 13.0 GPU, got {torch.version.cuda!r}")
    if torch.cuda.get_device_capability() < (12, 0):
        raise RuntimeError(f"expected SM120+, got {torch.cuda.get_device_capability()}")

    if len(sys.argv) == 2 and sys.argv[1] != "--skip-model":
        model_dir = Path(sys.argv[1])
        config = json.loads((model_dir / "config.json").read_text())
        quantization = config.get("quantization_config", {})
        if config.get("model_type") != "qwen3_5" or quantization.get("quant_method") != "fp8":
            raise RuntimeError("model is not Qwen3.8 FP8")
        if not (model_dir / "model.safetensors.index.json").is_file():
            raise RuntimeError("model weights are incomplete")
    elif len(sys.argv) > 2:
        raise SystemExit(f"usage: {sys.argv[0]} [--skip-model|MODEL_DIR]")

    print(json.dumps({"versions": versions, "gpu": torch.cuda.get_device_name()}))


if __name__ == "__main__":
    main()
