#!/bin/bash
set -e

export workspace="/opt/visual-perception-engine"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps/log}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
sudo chown $USER $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# Start MPS control daemon if not already running
if ! pgrep -x "nvidia-cuda-mps-control" > /dev/null; then
    echo "Starting CUDA MPS daemon..."
    nvidia-cuda-mps-control -d
fi

# If no command is passed, keep container alive
if [ $# -eq 0 ]; then
    echo "No command provided. Starting bash to keep the container running."
    exec /bin/bash
else
    exec "$@"
fi
