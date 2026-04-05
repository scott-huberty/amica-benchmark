#!/usr/bin/env bash
set -euo pipefail

RUNTIME="${RUNTIME:-docker}"
IMAGE_NAME="${1:-shuberty/amica:latest}"
APPTAINER_OUTPUT="${2:-amica.sif}"

case "${RUNTIME}" in
  docker)
    echo "Pulling Docker image ${IMAGE_NAME}"
    docker pull "${IMAGE_NAME}"
    ;;
  apptainer)
    echo "Pulling Apptainer image ${IMAGE_NAME} -> ${APPTAINER_OUTPUT}"
    apptainer pull "${APPTAINER_OUTPUT}" "docker://${IMAGE_NAME}"
    ;;
  *)
    echo "Unsupported runtime: ${RUNTIME}" >&2
    echo "Set RUNTIME=docker or RUNTIME=apptainer" >&2
    exit 1
    ;;
esac
