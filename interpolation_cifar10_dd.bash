#!/usr/bin/env bash
set -euo pipefail

# Interpolation inference (interpolation_h5.py, all pair modes) for every model
# of a width sweep. Runs are pulled from a shared queue (largest width first)
# by one worker per GPU; a fp32 k=64 run peaks at ~19.3 GiB, so do not put two
# large widths on one 3090.
#
# Run inside a container that also mounts the output disk, e.g.:
#   docker exec -d -w /workspace/jonal jonal-interp \
#     bash -c 'bash interpolation_cifar10_dd.bash > logs/interp_cifar10_dd.log 2>&1'
#
# Overrides: SWEEP_DIR=results/<sweep> GPUS="0 3" OUTPUT_ROOT=... EXTRA_ARGS="--epochs 0-10"
# A rerun resumes: interpolation_h5.py skips epoch blocks that are already written.
# WAIT_IDLE_MIB=1000: before each run a worker waits until its GPU uses less than
# this many MiB, so GPUs that are busy with other jobs join the sweep once free.
SWEEP_DIR="${SWEEP_DIR:-results/cifar10-dd-resnet18k-width-sweep-20261113T140649Z}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/ml-data/kobayashi/alpha_h5}"
GPUS=(${GPUS:-0 1 2 3})
NUM_PAIRS="${NUM_PAIRS:-1000}"
PAIR_SEED="${PAIR_SEED:-0}"
EXTRA_ARGS=(${EXTRA_ARGS:-})
WAIT_IDLE_MIB="${WAIT_IDLE_MIB:-}"

SWEEP_DIR="${SWEEP_DIR%/}"
SWEEP_NAME="$(basename "${SWEEP_DIR}")"
PAIRS_DIR="${PAIRS_DIR:-${OUTPUT_ROOT}/${SWEEP_NAME}/pairs}"
LOG_DIR="${LOG_DIR:-logs/interp-${SWEEP_NAME}}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# Run directories sorted by model width, largest first.
queue_file="${LOG_DIR}/.queue"
for run in "${SWEEP_DIR}"/*/; do
  [[ -f "${run}config.json" ]] || continue
  width="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_width"])' \
    "${run}config.json")"
  printf '%s\t%s\n' "${width}" "${run%/}"
done | sort -g -r -k1,1 | cut -f2 > "${queue_file}"
num_runs="$(wc -l < "${queue_file}")"

# Build the pairs once; every width then loads the same files from PAIRS_DIR.
# (Pairs depend only on the data, label noise seed and PAIR_SEED, not on the model.)
first_run="$(head -n 1 "${queue_file}")"
echo "$(date '+%F %T') building shared pairs in ${PAIRS_DIR}"
python3 interpolation_h5.py \
  --run-dir "${first_run}" \
  --gpu "${GPUS[0]}" \
  --num-pairs "${NUM_PAIRS}" \
  --pair-seed "${PAIR_SEED}" \
  --pairs-dir "${PAIRS_DIR}" \
  --pairs-only \
  > "${LOG_DIR}/pairs.log" 2>&1
cat "${LOG_DIR}/pairs.log"

next_run() {
  (
    flock 9
    local run
    run="$(head -n 1 "${queue_file}")"
    sed -i '1d' "${queue_file}"
    printf '%s' "${run}"
  ) 9>"${queue_file}.lock"
}

run_interpolation() {
  local gpu="$1"
  local run="$2"
  local name
  name="$(basename "${run}")"
  echo "$(date '+%F %T') start ${name} on GPU ${gpu}"
  if python3 interpolation_h5.py \
      --run-dir "${run}" \
      --gpu "${gpu}" \
      --num-pairs "${NUM_PAIRS}" \
      --pair-seed "${PAIR_SEED}" \
      --pairs-dir "${PAIRS_DIR}" \
      --output-root "${OUTPUT_ROOT}" \
      --experiment-name "${SWEEP_NAME}" \
      "${EXTRA_ARGS[@]}" \
      > "${LOG_DIR}/${name}.log" 2>&1; then
    echo "$(date '+%F %T') done  ${name}"
  else
    echo "$(date '+%F %T') FAILED ${name} (see ${LOG_DIR}/${name}.log)" >&2
    return 1
  fi
}

wait_for_idle_gpu() {
  local gpu="$1"
  [[ -n "${WAIT_IDLE_MIB}" ]] || return 0
  local used announced=0
  while used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)" \
      && (( used >= WAIT_IDLE_MIB )); do
    if (( ! announced )); then
      echo "$(date '+%F %T') GPU ${gpu} busy (${used} MiB), waiting"
      announced=1
    fi
    sleep 300
  done
}

worker() {
  local gpu="$1"
  local status=0
  local run
  while wait_for_idle_gpu "${gpu}" && run="$(next_run)" && [[ -n "${run}" ]]; do
    run_interpolation "${gpu}" "${run}" || status=1
  done
  return "${status}"
}

echo "Sweep:  ${SWEEP_DIR} (${num_runs} runs)"
echo "Output: ${OUTPUT_ROOT}/${SWEEP_NAME}"
echo "Logs:   ${LOG_DIR}"
echo "GPUs:   ${GPUS[*]}"

pids=()
for gpu in "${GPUS[@]}"; do
  worker "${gpu}" &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
rm -f "${queue_file}" "${queue_file}.lock"

if (( status != 0 )); then
  echo "One or more interpolation runs failed." >&2
  exit "${status}"
fi
echo "All interpolation runs completed."
