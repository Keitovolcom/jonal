#!/usr/bin/env bash
set -euo pipefail

# Model-wise double descent on CIFAR-10 (Nakkiran et al., 2019 style).
# resnet18k (Nakkiran et al.'s make_resnet18k, PyTorch default init) with
# width k uses channels [k, 2k, 4k, 8k], i.e. --model-width k / 64.
#
# Runs are pulled from a shared queue (largest width first) by
# SLOTS_PER_GPU workers on each GPU. Run inside the jonal-dev container:
#   docker exec -d -w /workspace/jonal jonal-dev \
#     bash -c 'bash train_cifar10_dd.bash > logs/cifar10_dd.log 2>&1'
#
# Overrides: GPUS="0 1 2 3" SLOTS_PER_GPU=2 WIDTHS="1 2 4 64" EPOCHS=1000 ...
SEED="${SEED:-42}"
MODEL="${MODEL:-resnet18k}"  # Nakkiran et al. make_resnet18k
EPOCHS="${EPOCHS:-1000}"
DATASET="cifar10"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.0001}"
OPTIMIZER="${OPTIMIZER:-adam}"
MOMENTUM="${MOMENTUM:-0.0}"
LOSS="${LOSS:-crossentropy}"
LABEL_NOISE="${LABEL_NOISE:-0.2}"
AUGMENTATION="${AUGMENTATION:-default}"  # RandomCrop(pad 4) + HorizontalFlip
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
GPUS=(${GPUS:-0 1 2 3})
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
WIDTHS=(${WIDTHS:-64 56 48 40 32 28 24 20 18 16 14 12 11 10 9 8 7 6 5 4 3 2 1})
WANDB_PROJECT="${WANDB_PROJECT:-kobayashi_cifar10}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-dsml-kernel24}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_NAME="cifar10-dd-${MODEL}-width-sweep-${RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${SWEEP_NAME}}"
LOG_DIR="${LOG_DIR:-logs/${SWEEP_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-${SWEEP_NAME}}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
queue_file="${LOG_DIR}/.queue"
printf '%s\n' "${WIDTHS[@]}" > "${queue_file}"

wandb_args=(
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-group "${WANDB_GROUP}"
)
if [[ -n "${WANDB_ENTITY}" ]]; then
  wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
fi

# Pop the first width from the queue; prints nothing when it is empty.
next_width() {
  (
    flock 9
    local width
    width="$(head -n 1 "${queue_file}")"
    sed -i '1d' "${queue_file}"
    printf '%s' "${width}"
  ) 9>"${queue_file}.lock"
}

run_experiment() {
  local gpu="$1"
  local k="$2"
  local multiplier
  multiplier="$(awk -v k="${k}" 'BEGIN { printf "%.10g", k / 64 }')"

  echo "$(date '+%F %T') start k=${k} (multiplier=${multiplier}) on GPU ${gpu}"
  if python3 train.py \
      --model "${MODEL}" \
      --model-width "${multiplier}" \
      --dataset "${DATASET}" \
      --lr "${LR}" \
      --batch-size "${BATCH_SIZE}" \
      --gpu "${gpu}" \
      --num-workers "${NUM_WORKERS}" \
      --augmentation "${AUGMENTATION}" \
      --label-noise "${LABEL_NOISE}" \
      --optimizer "${OPTIMIZER}" \
      --momentum "${MOMENTUM}" \
      --loss "${LOSS}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
      --checkpoint-model-only \
      --output-dir "${OUTPUT_DIR}" \
      "${wandb_args[@]}" \
      > "${LOG_DIR}/k-${k}.log" 2>&1; then
    echo "$(date '+%F %T') done  k=${k}"
  else
    echo "$(date '+%F %T') FAILED k=${k} (see ${LOG_DIR}/k-${k}.log)" >&2
    return 1
  fi
}

worker() {
  local gpu="$1"
  local status=0
  local k
  while k="$(next_width)" && [[ -n "${k}" ]]; do
    run_experiment "${gpu}" "${k}" || status=1
  done
  return "${status}"
}

echo "Results: ${OUTPUT_DIR}"
echo "Logs:    ${LOG_DIR}"
echo "Widths:  ${WIDTHS[*]}"

pids=()
for slot in $(seq 1 "${SLOTS_PER_GPU}"); do
  for gpu in "${GPUS[@]}"; do
    worker "${gpu}" &
    pids+=($!)
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
rm -f "${queue_file}" "${queue_file}.lock"

if (( status != 0 )); then
  echo "One or more experiments failed." >&2
  exit "${status}"
fi
echo "All width experiments completed."
