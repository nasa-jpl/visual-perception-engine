#!/usr/bin/env bash

set -u

# Run this script from anywhere. It moves into the benchmarking directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONTAINER="${CONTAINER:-engine}"
INTERVAL_SEC="${INTERVAL_SEC:-0.1}"
OUTPUT_FILE="${OUTPUT_FILE:-results_cpu_utilization.jsonl}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

echo "Benchmarking directory: ${SCRIPT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Container: ${CONTAINER}"
echo "CPU sampling interval: ${INTERVAL_SEC} sec (10 Hz)"
echo "Output JSONL: ${OUTPUT_FILE}"
echo "Continue on error: ${CONTINUE_ON_ERROR}"

# Make sure that Jetson is set to maximum power mode.
sudo nvpmodel -m 0

# Make sure that fan is set to maximum speed.
sudo jetson_clocks --fan

# Make sure CUDA MPS is on.
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps/log

# Make sure that the directories exist and have correct permissions.
sudo mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
sudo chown "$(id -un)" "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

# Start the daemon. If it is already running, this may print a warning, which is fine.
nvidia-cuda-mps-control -d || true

frameworks=(
  vp_engine
  fully_sequential_torch
  fully_sequential_tensorrt
  head_sequential_torch
  head_sequential_tensorrt
  merged_torch
  merged_tensorrt
)

failed_runs=()

run_one_benchmark() {
  local framework="$1"
  local num_heads="$2"
  local obj_det="$3"

  local tag="${framework}_heads${num_heads}"
  local cmd=("${PYTHON_BIN}" "run_with_gpu_monitoring.py" "--container" "${CONTAINER}" "--framework" "${framework}" "--num_heads" "${num_heads}")

  if [[ "${obj_det}" == "1" ]]; then
    tag="${tag}_objdet"
    cmd+=("--obj_det_heads")
  else
    tag="${tag}_no_objdet"
  fi

  echo
  echo "================================================================================"
  echo "Running benchmark tag: ${tag}"
  echo "Command: ${cmd[*]}"
  echo "================================================================================"

  "${PYTHON_BIN}" run_with_cpu_core_monitoring.py \
    --output-file "${OUTPUT_FILE}" \
    --tag "${tag}" \
    --interval-sec "${INTERVAL_SEC}" \
    -- \
    "${cmd[@]}"

  local exit_code=$?

  if [[ ${exit_code} -ne 0 ]]; then
    echo "Benchmark failed: ${tag} with exit code ${exit_code}"
    failed_runs+=("${tag}:exit_${exit_code}")

    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      echo "Stopping because CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
      exit "${exit_code}"
    fi
  fi
}

for framework in "${frameworks[@]}"; do
  for num_heads in {1..8}; do

    if [[ "${framework}" == merged_* ]]; then
      # Merged models: run once, without object-detection flag.
      run_one_benchmark "${framework}" "${num_heads}" "0"
    else
      # Non-merged models: run twice, without and with object-detection heads.
      run_one_benchmark "${framework}" "${num_heads}" "0"
      run_one_benchmark "${framework}" "${num_heads}" "1"
    fi

  done
done

echo
echo "================================================================================"
echo "CPU core benchmark batch completed."
echo "Results saved in: ${SCRIPT_DIR}/${OUTPUT_FILE}"
echo "================================================================================"

if [[ ${#failed_runs[@]} -gt 0 ]]; then
  echo
  echo "Failed runs:"
  for failed in "${failed_runs[@]}"; do
    echo "  - ${failed}"
  done
  exit 1
fi

exit 0