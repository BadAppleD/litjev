#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_ID=${LITJEV_MODEL_ID:-Qwen/Qwen3.8-27B-FP8}
MODEL_REVISION=${LITJEV_MODEL_REVISION:-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}
MODEL_DIR=${LITJEV_MODEL_DIR:-"$ROOT/models/Qwen3.8-27B-FP8"}

command -v hf >/dev/null || { echo "missing command: hf" >&2; exit 1; }
mkdir -p "$MODEL_DIR"
hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
hf cache verify "$MODEL_ID" --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR" --fail-on-missing-files
echo "downloaded: $MODEL_DIR"
