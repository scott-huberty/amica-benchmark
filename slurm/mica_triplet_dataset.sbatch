#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [ "$#" -lt 14 ]; then
  echo "Usage: $0 DATASET_SET RUN_DIR MAX_ITER THREADS FORTRAN_IMAGE N_MIXTURES CONTAINER_RUNTIME APPTAINER_IMAGE RANDOM_STATE DEVICE VERBOSE BENCH_ROOT BENCH_REPO DAAREM_ACCELERATOR_ORDER [N_COMPONENTS]"
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
RANDOM_STATE="$9"
shift 9
DEVICE="$1"
VERBOSE="$2"
BENCH_ROOT="$3"
BENCH_REPO="$4"
DAAREM_ACCELERATOR_ORDER="$5"
N_COMPONENTS="${6:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -n "${CONDA_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
  module load "${CONDA_MODULE}"
fi
if [ -n "${CONDA_ENV:-}" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [ -n "${CONDA_SH:-}" ] && [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
  else
    echo "CONDA_ENV=${CONDA_ENV} was set, but no conda command or CONDA_SH was found."
    exit 1
  fi
  conda activate "${CONDA_ENV}"
fi

if [ "${CONTAINER_RUNTIME}" = "apptainer" ] && command -v module >/dev/null 2>&1; then
  module load "${APPTAINER_MODULE:-apptainer}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"

FORTRAN_RUN_DIR="${RUN_DIR}/fortran"
EM_RUN_DIR="${RUN_DIR}/python_em"
DAAREM_RUN_DIR="${RUN_DIR}/python_daarem"
mkdir -p "${FORTRAN_RUN_DIR}" "${EM_RUN_DIR}" "${DAAREM_RUN_DIR}"

FORTRAN_CMD=(
  "${PYTHON_BIN}" "${BENCH_REPO}/scripts/run_mica_fortran.py"
  --dataset-set "${DATASET_SET}"
  --run-dir "${FORTRAN_RUN_DIR}"
  --max-iter "${MAX_ITER}"
  --n-mixtures "${N_MIXTURES}"
  --container-runtime "${CONTAINER_RUNTIME}"
  --fortran-image "${FORTRAN_IMAGE}"
  --fortran-threads "${THREADS}"
)
if [ -n "${APPTAINER_IMAGE}" ]; then
  FORTRAN_CMD+=(--apptainer-image "${APPTAINER_IMAGE}")
fi
if [ -n "${N_COMPONENTS}" ]; then
  FORTRAN_CMD+=(--n-components "${N_COMPONENTS}")
fi

PYTHON_COMMON=(
  --dataset-set "${DATASET_SET}"
  --fortran-out "${FORTRAN_RUN_DIR}/fortran_out"
  --bench-root "${BENCH_ROOT}"
  --max-iter "${MAX_ITER}"
  --n-mixtures "${N_MIXTURES}"
  --random-state "${RANDOM_STATE}"
  --device "${DEVICE}"
  --verbose "${VERBOSE}"
  --python-threads "${THREADS}"
)
if [ -n "${N_COMPONENTS}" ]; then
  PYTHON_COMMON+=(--n-components "${N_COMPONENTS}")
fi

EM_CMD=(
  "${PYTHON_BIN}" "${BENCH_REPO}/scripts/run_mica_python.py"
  "${PYTHON_COMMON[@]}"
  --run-dir "${EM_RUN_DIR}"
  --optimizer em
)

DAAREM_CMD=(
  "${PYTHON_BIN}" "${BENCH_REPO}/scripts/run_mica_python.py"
  "${PYTHON_COMMON[@]}"
  --run-dir "${DAAREM_RUN_DIR}"
  --optimizer daarem
)
if [ -n "${DAAREM_ACCELERATOR_ORDER}" ]; then
  DAAREM_CMD+=(--accelerator-order "${DAAREM_ACCELERATOR_ORDER}")
fi

echo "Running Fortran AMICA: ${FORTRAN_CMD[*]}"
"${FORTRAN_CMD[@]}"

echo "Running AMICA-Python EM: ${EM_CMD[*]}"
"${EM_CMD[@]}"

echo "Running AMICA-Python DAAREM: ${DAAREM_CMD[*]}"
"${DAAREM_CMD[@]}"
