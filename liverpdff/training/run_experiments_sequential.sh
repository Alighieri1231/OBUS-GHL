#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MAX_EPOCHS="${MAX_EPOCHS:-1}"
DEVICES="${DEVICES:-2}"
ACCELERATOR="${ACCELERATOR:-gpu}"
STRATEGY="${STRATEGY:-ddp}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
WANDB_MODE="${WANDB_MODE:-offline}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CONFIGS=("$@")
if [ "${#CONFIGS[@]}" -eq 0 ]; then
  CONFIGS=(
    "liverpdff/training/configs/pdff_additive_attention.yaml"
    "liverpdff/training/configs/pdff_milattention.yaml"
    "liverpdff/training/configs/pdff_lstm.yaml"
    "liverpdff/training/configs/pdff_meanpool.yaml"
    "liverpdff/training/configs/pdff_convlstm.yaml"
  )
fi

for config in "${CONFIGS[@]}"; do
  echo "Running $config"
  WANDB_MODE="$WANDB_MODE" PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF" PYTHONNOUSERSITE=1 \
    python -m liverpdff.training.train fit \
    --config "$config" \
    --trainer.max_epochs "$MAX_EPOCHS" \
    --trainer.min_epochs 1 \
    --trainer.accelerator "$ACCELERATOR" \
    --trainer.devices "$DEVICES" \
    --trainer.strategy "$STRATEGY" \
    --trainer.check_val_every_n_epoch "$CHECK_VAL_EVERY_N_EPOCH" \
    --trainer.num_sanity_val_steps 0 \
    --trainer.limit_train_batches "$LIMIT_TRAIN_BATCHES" \
    --trainer.limit_val_batches "$LIMIT_VAL_BATCHES" \
    --trainer.use_distributed_sampler true
done
