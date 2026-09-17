#!/usr/bin/env bash
#SBATCH --job-name=litjev-sequential10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
nvidia-smi
"${UV_BIN:-uv}" run --locked --no-sync python scripts/run_cluster_benchmark.py \
  --model "${LITJEV_MODEL:?Set LITJEV_MODEL to a Qwen checkpoint path or model ID}" \
  --sequential --output "results/job-${SLURM_JOB_ID}.json"
