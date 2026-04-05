#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [ "$#" -lt 9 ]; then
  echo "Usage: $0 DATASET_SET RUN_DIR MAX_ITER THREADS FORTRAN_IMAGE N_MIXTURES CONTAINER_RUNTIME APPTAINER_IMAGE BENCH_REPO [N_COMPONENTS]"
  exit 1
fi

DATASET_SET="$1"
RUN_DIR="$2"
MAX_ITER="$3"
THREADS="$4"
FORTRAN_IMAGE="$5"
N_MIXTURES="$6"
CONTAINER_RUNTIME="$7"
APPTAINER_IMAGE="$8"
BENCH_REPO="$9"
N_COMPONENTS="${10:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"

mkdir -p "${RUN_DIR}"

CMD=(
  "${PYTHON_BIN}" "${BENCH_REPO}/scripts/run_mica_fortran.py"
  --dataset-set "${DATASET_SET}"
  --run-dir "${RUN_DIR}"
  --max-iter "${MAX_ITER}"
  --n-mixtures "${N_MIXTURES}"
  --container-runtime "${CONTAINER_RUNTIME}"
  --fortran-image "${FORTRAN_IMAGE}"
  --fortran-threads "${THREADS}"
)
if [ -n "${APPTAINER_IMAGE}" ]; then
  CMD+=(--apptainer-image "${APPTAINER_IMAGE}")
fi
if [ -n "${N_COMPONENTS}" ]; then
  CMD+=(--n-components "${N_COMPONENTS}")
fi

"${CMD[@]}"
