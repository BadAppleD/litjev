#!/usr/bin/env bash
#SBATCH --job-name=litjev-collect
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out
# Collect decision-head records: fast readout features plus slow-path outcomes.
# Shard across array tasks with --array=0-N; each task takes LITJEV_SHARD questions.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
nvidia-smi
SHARD="${LITJEV_SHARD:-500}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
"${UV_BIN:-uv}" run --locked --no-sync litjev-collect \
  --model "${LITJEV_MODEL:?Set LITJEV_MODEL to a Qwen checkpoint path or model ID}" \
  --split "${LITJEV_SPLIT:-test}" \
  --layers "${LITJEV_LAYERS:--1}" \
  --budget "${LITJEV_BUDGET:-512}" \
  --offset $((TASK * SHARD)) \
  --limit "$SHARD" \
  --device-map cuda:0 \
  --output "results/records-${SLURM_JOB_ID}-${TASK}.npz"
