#!/usr/bin/env bash
set -euo pipefail

# Environment variables can override these values:
#   GPU2=2 GPU3=3 WIDTH_MULTIPLIER=0.5 CHECKPOINT_INTERVAL=0 WANDB_MODE=online bash train_labelnoise.bash
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-0}"
WIDTH_MULTIPLIER="${WIDTH_MULTIPLIER:-0.5}"
WANDB_PROJECT="${WANDB_PROJECT:-kobayashi_syuron}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-dsml-kernel24}"
WANDB_GROUP="${WANDB_GROUP:-}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"

# Keep each label-noise sweep separate so runs do not overwrite each other.
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/cnn5layer-labelnoise-sweep-${RUN_ID}}"

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
  local label_noise="$2"

  echo "Starting label_noise=${label_noise}, width=${WIDTH_MULTIPLIER} on GPU ${gpu}"
  python3 train.py \
    --model cnn5layer \
    --model-width "${WIDTH_MULTIPLIER}" \
    --dataset emnist-digits \
    --lr 0.01 \
    --batch-size 256 \
    --gpu "${gpu}" \
    --label-noise "${label_noise}" \
    --optimizer sgd \
    --loss crossentropy \
    --epochs 2000 \
    --seed 42 \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
    --output-dir "${OUTPUT_DIR}" \
    "${wandb_args[@]}"
}

run_worker() {
  local gpu="$1"
  shift

  while (( $# > 0 )); do
    run_experiment "${gpu}" "$1"
    shift
  done
}

mkdir -p "${OUTPUT_DIR}"
echo "Results will be saved under ${OUTPUT_DIR}"

# Run two label-noise conditions per GPU in parallel.
run_worker "${GPU2}" 0.10 0.08 &
gpu2_pid=$!
run_worker "${GPU3}" 0.05 0.03 &
gpu3_pid=$!

status=0
if ! wait "${gpu2_pid}"; then
  status=1
fi
if ! wait "${gpu3_pid}"; then
  status=1
fi

if (( status != 0 )); then
  echo "One or more experiments failed." >&2
  exit "${status}"
fi

echo "All label-noise experiments completed."
