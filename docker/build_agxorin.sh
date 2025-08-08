#!/bin/bash
set -euo pipefail

export _UID="$(id -u)"
export _GID="$(id -g)"
export USER="${USER:-$(whoami)}"
export DISPLAY=0
export WORKSPACE="/opt/visual-perception-engine"
export REPOS="/opt/visual-perception-engine"
export DISPLAY="${DISPLAY:-:0}"

if [ -z "${SSH_AUTH_SOCK:-}" ]; then
  eval "$(ssh-agent -s)" >/dev/null
fi

#we'll build torch2trt wheel if needed. This works as it fails in Dockerfile.
if [ ! -f ./torch2trt-0.5.0-py3-none-any.whl ]; then
  source ./torch2trt_function.sh
  build_torch2trt
fi

#Build image
docker compose -f docker-compose-agxorin.yml build
# --no-cache

#Create container from image.
# docker compose -f docker-compose-agxorin.yml up -d

#container_name=$(docker compose ps -q | head -n1)
