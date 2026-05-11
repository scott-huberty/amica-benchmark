#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [ "$#" -lt 12 ]; then
  echo "Usage: $0 DATASET_SET RUN_DIR MAX_ITER THREADS OPTIMIZER FORTRAN_SEARCH_ROOT N_MIXTURES RANDOM_STATE DEVICE VERBOSE BENCH_ROOT BENCH_REPO [N_COMPONENTS]"
  echo "Optional: set AMICA_ACCELERATOR_ORDER to override accelerator_order."
  exit 1
fi

DATASET_SET="$1"
RUN_DIR="$2"
MAX_ITER="$3"
THREADS="$4"
OPTIMIZER="$5"
FORTRAN_SEARCH_ROOT="$6"
N_MIXTURES="$7"
RANDOM_STATE="$8"
DEVICE="$9"
shift 9
VERBOSE="$1"
shift 1
BENCH_ROOT="$1"
shift 1
BENCH_REPO="$1"
shift 1
N_COMPONENTS="${1:-}"

if [ "${OPTIMIZER}" != "em" ] && [ "${OPTIMIZER}" != "daarem" ]; then
  echo "Unsupported optimizer '${OPTIMIZER}'. Expected 'em' or 'daarem'."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"

mkdir -p "${RUN_DIR}"

CMD=(
  "${PYTHON_BIN}" "${BENCH_REPO}/scripts/run_mica_python.py"
  --dataset-set "${DATASET_SET}"
  --fortran-search-root "${FORTRAN_SEARCH_ROOT}"
  --run-dir "${RUN_DIR}"
  --bench-root "${BENCH_ROOT}"
  --max-iter "${MAX_ITER}"
  --n-mixtures "${N_MIXTURES}"
  --random-state "${RANDOM_STATE}"
  --device "${DEVICE}"
  --optimizer "${OPTIMIZER}"
  --verbose "${VERBOSE}"
  --python-threads "${THREADS}"
)
if [ -n "${AMICA_ACCELERATOR_ORDER:-}" ]; then
  CMD+=(--accelerator-order "${AMICA_ACCELERATOR_ORDER}")
fi
if [ -n "${N_COMPONENTS}" ]; then
  CMD+=(--n-components "${N_COMPONENTS}")
fi

"${CMD[@]}"
