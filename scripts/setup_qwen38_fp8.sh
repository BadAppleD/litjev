#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${LITJEV_ENV_DIR:-"$ROOT/.venv-qwen38"}
CACHE_DIR=${LITJEV_CACHE_DIR:-"$ROOT/.cache-qwen38"}
MODEL_ID=${LITJEV_MODEL_ID:-Qwen/Qwen3.8-27B-FP8}
MODEL_REVISION=${LITJEV_MODEL_REVISION:-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}
MODEL_DIR=${LITJEV_MODEL_DIR:-"$ROOT/models/Qwen3.8-27B-FP8"}
CAUSAL_COMMIT=cd81f0413cad2fc1e6f17e785ac39f59aae690cd
SKIP_MODEL=0

if [[ ${1:-} == "--skip-model" ]]; then
  SKIP_MODEL=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--skip-model]" >&2
  exit 2
fi

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  echo "requires Linux x86_64" >&2; exit 1;
}
for command in uv g++ nvcc nvidia-smi; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 1; }
done
nvcc --version | grep -q 'release 13\.' || {
  echo "CUDA 13.x toolkit is required to build the SM120 kernel" >&2; exit 1;
}

mkdir -p "$CACHE_DIR/source" "$CACHE_DIR/wheels" "$MODEL_DIR"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv --python 3.12 "$ENV_DIR"
fi
PYTHON="$ENV_DIR/bin/python"
uv pip sync --python "$PYTHON" --index-strategy unsafe-best-match --require-hashes \
  "$ROOT/requirements-qwen38-cu130.lock"

CAUSAL_WHEEL=$(find "$CACHE_DIR/wheels" -maxdepth 1 \
  -name 'causal_conv1d-1.7.0+sm120-*-linux_x86_64.whl' -print -quit)
if [[ -z "$CAUSAL_WHEEL" ]]; then
  command -v git >/dev/null || { echo "missing command: git" >&2; exit 1; }
  CAUSAL_SOURCE="$CACHE_DIR/source/causal-conv1d-$CAUSAL_COMMIT"
  if [[ ! -d "$CAUSAL_SOURCE/.git" ]]; then
    git clone --filter=blob:none --no-checkout \
      https://github.com/Dao-AILab/causal-conv1d.git "$CAUSAL_SOURCE"
  fi
  git -C "$CAUSAL_SOURCE" fetch --depth 1 origin "$CAUSAL_COMMIT"
  git -C "$CAUSAL_SOURCE" checkout --detach "$CAUSAL_COMMIT"
  CAUSAL_CONV1D_FORCE_BUILD=TRUE \
  CAUSAL_CONV1D_LOCAL_VERSION=sm120 \
  TORCH_CUDA_ARCH_LIST=12.0 \
    "$PYTHON" -m build --wheel --no-isolation \
      --outdir "$CACHE_DIR/wheels" "$CAUSAL_SOURCE"
  CAUSAL_WHEEL=$(find "$CACHE_DIR/wheels" -maxdepth 1 \
    -name 'causal_conv1d-1.7.0+sm120-*-linux_x86_64.whl' -print -quit)
fi
[[ -n "$CAUSAL_WHEEL" ]] || { echo "causal-conv1d wheel was not built" >&2; exit 1; }

uv pip install --python "$PYTHON" --no-deps "$CAUSAL_WHEEL"
uv pip install --python "$PYTHON" --no-deps --editable "$ROOT"

VERIFY_ARG=$MODEL_DIR
if [[ $SKIP_MODEL -eq 0 ]]; then
  PATH="$ENV_DIR/bin:$PATH" \
  LITJEV_MODEL_ID="$MODEL_ID" \
  LITJEV_MODEL_REVISION="$MODEL_REVISION" \
  LITJEV_MODEL_DIR="$MODEL_DIR" \
    "$ROOT/scripts/download_qwen38_fp8.sh"
else
  VERIFY_ARG=--skip-model
fi

"$PYTHON" "$ROOT/scripts/verify_qwen38_env.py" "$VERIFY_ARG"
PYTHONPATH="$ROOT/src" "$PYTHON" -m unittest discover \
  -s "$ROOT/tests" -p 'test_fp8_loading.py'

echo "ready: $ENV_DIR/bin/litjev-serve --model $MODEL_DIR --device-map cuda:0 --eager-load"
