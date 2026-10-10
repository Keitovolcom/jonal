#!/usr/bin/env bash
set -euo pipefail

# Environment variables can override these values:
#   GPU0=0 GPU1=1 CHECKPOINT_INTERVAL=5 WANDB_MODE=online bash train_seeds.bash
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-kobayashi_syuron}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-dsml-kernel24}"
WANDB_GROUP="${WANDB_GROUP:-}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

# Keep every seed sweep separate because train.py does not overwrite results.
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/cnn5layer-seed-sweep-${RUN_ID}}"

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

run_experiment() {
  local gpu="$1"
  local seed="$2"

  echo "Starting seed=${seed} on GPU ${gpu}"
  python3 train.py \
    --model cnn5layer \
    --dataset emnist-digits \
    --lr 0.01 \
    --batch-size 256 \
    --gpu "${gpu}" \
    --label-noise 0.2 \
    --optimizer sgd \
    --loss crossentropy \
    --epochs 2000 \
    --seed "${seed}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
    --output-dir "${OUTPUT_DIR}" \
    "${wandb_args[@]}"
}

run_worker() {
  local gpu="$1"
  shift

  for seed in "$@"; do
    run_experiment "${gpu}" "${seed}"
  done
}

mkdir -p "${OUTPUT_DIR}"
echo "Results will be saved under ${OUTPUT_DIR}"

# Run two seeds sequentially on each GPU. The two workers run in parallel.
run_worker "${GPU0}" 43 45 &
gpu0_pid=$!
run_worker "${GPU1}" 44 46 &
gpu1_pid=$!

status=0
if ! wait "${gpu0_pid}"; then
  status=1
fi
if ! wait "${gpu1_pid}"; then
  status=1
fi

if (( status != 0 )); then
  echo "One or more experiments failed." >&2
  exit "${status}"
fi

echo "All seed experiments completed."
