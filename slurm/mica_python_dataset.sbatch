#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [ "$#" -lt 11 ]; then
  echo "Usage: $0 DATASET_SET RUN_DIR MAX_ITER THREADS FORTRAN_SEARCH_ROOT N_MIXTURES RANDOM_STATE DEVICE VERBOSE BENCH_ROOT BENCH_REPO [N_COMPONENTS]"
  exit 1
fi

DATASET_SET="$1"
RUN_DIR="$2"
MAX_ITER="$3"
THREADS="$4"
FORTRAN_SEARCH_ROOT="$5"
N_MIXTURES="$6"
RANDOM_STATE="$7"
DEVICE="$8"
VERBOSE="$9"
shift 9
BENCH_ROOT="$1"
shift 1
BENCH_REPO="$1"
shift 1
N_COMPONENTS="${1:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export OPM_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"

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
  --verbose "${VERBOSE}"
  --python-threads "${THREADS}"
)
if [ -n "${N_COMPONENTS}" ]; then
  CMD+=(--n-components "${N_COMPONENTS}")
fi

"${CMD[@]}"
