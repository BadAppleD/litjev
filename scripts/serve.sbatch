#!/usr/bin/env bash
#SBATCH --job-name=litjev-playground
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=results/serve-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Loopback only: use an SSH jump-host tunnel directly to the allocated compute node.
exec "${UV_BIN:-uv}" run --locked --no-sync litjev \
  --model "${LITJEV_MODEL:?Set LITJEV_MODEL to a Qwen checkpoint path or model ID}" \
  --device-map cuda:0 --port 8800
