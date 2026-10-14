#!/usr/bin/env bash
set -euo pipefail

# Interpolation inference for all pair modes, followed by the instability metrics.
#   RUN_DIR=results/<experiment_name> NUM_PAIRS=1000 GPU=0 bash interpolation.bash
# Optional: EPOCHS='0-100' PHASE_EPOCHS='30,60' OUTPUT_ROOT=interpolation
RUN_DIR="${RUN_DIR:?set RUN_DIR to a train.py result directory}"
NUM_PAIRS="${NUM_PAIRS:-1000}"
PAIR_SEED="${PAIR_SEED:-0}"
GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-interpolation}"
PAIRS_PER_CHUNK="${PAIRS_PER_CHUNK:-100}"
EPOCHS="${EPOCHS:-}"
PHASE_EPOCHS="${PHASE_EPOCHS:-}"
MODES=(${MODES:-clean-noisy clean-clean test-test})

experiment_name="$(basename "${RUN_DIR%/}")"
epoch_args=()
if [[ -n "${EPOCHS}" ]]; then
  epoch_args+=(--epochs "${EPOCHS}")
fi
phase_args=()
if [[ -n "${PHASE_EPOCHS}" ]]; then
  phase_args+=(--phase-epochs "${PHASE_EPOCHS}")
fi

mode_dirs=()
for mode in "${MODES[@]}"; do
  echo "Starting mode=${mode}, pairs=${NUM_PAIRS} on GPU ${GPU}"
  python3 interpolation.py \
    --run-dir "${RUN_DIR}" \
    --mode "${mode}" \
    --num-pairs "${NUM_PAIRS}" \
    --pair-seed "${PAIR_SEED}" \
    --gpu "${GPU}" \
    --pairs-per-chunk "${PAIRS_PER_CHUNK}" \
    --output-root "${OUTPUT_ROOT}" \
    "${epoch_args[@]}"
  mode_dirs+=("${OUTPUT_ROOT}/${experiment_name}/${mode}")
done

python3 interpolation_metrics.py --mode-dir "${mode_dirs[@]}" "${phase_args[@]}"
echo "All interpolation experiments completed."
