#!/usr/bin/env bash
set -euo pipefail

# Environment variables can override these values:
#   GPU=0 CHECKPOINT_INTERVAL=10 WANDB_MODE=offline bash train_cifar10.bash
#   GRAY_SCALE=true bash train_cifar10.bash
SEED="${SEED:-42}"
MODEL="${MODEL:-preactresnet18}"
EPOCHS="${EPOCHS:-2000}"
DATASET="cifar10"
GRAY_SCALE="${GRAY_SCALE:-false}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.0001}"
OPTIMIZER="${OPTIMIZER:-adam}"
MOMENTUM="${MOMENTUM:-0.0}"
LOSS="${LOSS:-crossentropy}"
LABEL_NOISE="${LABEL_NOISE:-0.2}"
GPU="${GPU:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-kobayashi_cifar10}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-dsml-kernel24}"
WANDB_GROUP="${WANDB_GROUP:-}"

# Width multipliers relative to the standard [64, 128, 256, 512] channels.
# Nakkiran-style ResNet18 width k corresponds to a multiplier of k / 64.
MODEL_WIDTHS=(1.0)

extra_args=()
if [[ "${GRAY_SCALE}" == true ]]; then
  extra_args+=(--input-channels 1)
fi

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

# Noisy labels are saved under data/cifar10/ on the first run and reused
# automatically afterwards, so no flag is needed for that.
for model_width in "${MODEL_WIDTHS[@]}"; do
  echo "Starting ${MODEL} width=${model_width} on GPU ${GPU}"
  python3 train.py \
    --model "${MODEL}" \
    --model-width "${model_width}" \
    --dataset "${DATASET}" \
    --lr "${LR}" \
    --batch-size "${BATCH_SIZE}" \
    --gpu "${GPU}" \
    --num-workers "${NUM_WORKERS}" \
    --label-noise "${LABEL_NOISE}" \
    --optimizer "${OPTIMIZER}" \
    --momentum "${MOMENTUM}" \
    --loss "${LOSS}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
    "${extra_args[@]}" \
    "${wandb_args[@]}"
done
