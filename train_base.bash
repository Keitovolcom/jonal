#!/usr/bin/env bash
set -euo pipefail

# Environment variables can override these values:
#   CHECKPOINT_INTERVAL=5 WANDB_MODE=online bash train.bash
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-kobayashi_syuron}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-dsml-kernel24}"
WANDB_GROUP="${WANDB_GROUP:-}"

wandb_args=(
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
)
if [[ -n "${WANDB_ENTITY}" ]]; then
  wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_GROUP}" ]]; then
  wandb_args+=(--wandb-group "${WANDB_GROUP}")
fi

python3 train.py \
  --model cnn5layer \
  --dataset emnist-digits \
  --lr 0.01 \
  --batch-size 256 \
  --gpu 1 \
  --label-noise 0.2 \
  --optimizer sgd \
  --loss crossentropy \
  --epochs 2000 \
  --seed 42 \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  "${wandb_args[@]}"
