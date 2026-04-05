#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-shuberty/amica:latest}"

echo "Pulling ${IMAGE_NAME}"
docker pull "${IMAGE_NAME}"
