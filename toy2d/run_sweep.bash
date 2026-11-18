#!/usr/bin/env bash
# Toy 2-D EDD sweep. Runs are distributed round-robin over GPUs, JOBS_PER_GPU at a time.
#   GPUS="0 1 2 3" JOBS_PER_GPU=7 EPOCHS=20000 bash toy2d/run_sweep.bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=(${GPUS:-0 1 2 3})
JOBS_PER_GPU="${JOBS_PER_GPU:-7}"
EPOCHS="${EPOCHS:-20000}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/toy2d-${RUN_ID}}"
SEEDS=(${SEEDS:-0 1 2})
mkdir -p "${OUTPUT_DIR}/logs"

# dim sigma noise lr.  COND_SET=extra selects spec conditions 4/5.
if [[ "${COND_SET:-main}" == "extra" ]]; then
CONDITIONS=(
  "2 0 0.5 1e-3"     # cond 4
  "50 0.3 0.5 1e-3"
  "50 0.1 0.2 1e-3"  # cond 5
  "50 1.0 0.2 1e-3"
)
else
CONDITIONS=(
  "2 0 0.0 1e-3"     # cond 1
  "2 0 0.2 1e-3"     # cond 2
  "50 0.3 0.2 1e-3"  # cond 3
  "10 0.3 0.2 1e-3"  # dim sweep
  "200 0.3 0.2 1e-3"
  "50 0.3 0.0 1e-3"  # d=50 clean control
  "2 0 0.2 1e-4"     # lower lr (spec fallback)
  "10 0.3 0.2 1e-4"
  "50 0.3 0.2 1e-4"
)
fi

jobs_list=()
for c in "${CONDITIONS[@]}"; do
  for s in "${SEEDS[@]}"; do jobs_list+=("$c $s"); done
done

worker() {
  local gpu="$1"; shift
  for j in "$@"; do
    read -r dim sigma noise lr seed <<<"${j}"
    local tag="d-${dim}_sigma-${sigma}_noise-${noise}_lr-${lr}_seed-${seed}"
    echo "[$(date -u +%T)] GPU ${gpu} start ${tag}"
    python3 toy2d/train_toy.py --dim "${dim}" --sigma "${sigma}" --label-noise "${noise}" \
      --lr "${lr}" --seed "${seed}" --epochs "${EPOCHS}" --gpu "${gpu}" \
      --output-dir "${OUTPUT_DIR}" >"${OUTPUT_DIR}/logs/${tag}.log" 2>&1 \
      && echo "[$(date -u +%T)] GPU ${gpu} done  ${tag}" \
      || echo "[$(date -u +%T)] GPU ${gpu} FAIL  ${tag}"
  done
}

nslots=$(( ${#GPUS[@]} * JOBS_PER_GPU ))
pids=()
for ((k = 0; k < nslots; k++)); do
  mine=()
  for ((i = k; i < ${#jobs_list[@]}; i += nslots)); do mine+=("${jobs_list[i]}"); done
  (( ${#mine[@]} )) || continue
  worker "${GPUS[k % ${#GPUS[@]}]}" "${mine[@]}" &
  pids+=($!)
done
echo "Launched ${#jobs_list[@]} runs in ${#pids[@]} slots -> ${OUTPUT_DIR}"
wait
echo "All runs finished."
