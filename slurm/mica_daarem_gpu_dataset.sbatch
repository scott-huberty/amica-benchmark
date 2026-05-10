#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [ "$#" -lt 25 ]; then
  echo "Usage: $0 DATASET_SET RUN_DIR MAX_ITER THREADS FORTRAN_SEARCH_ROOT N_MIXTURES RANDOM_STATE DEVICE VERBOSE BENCH_ROOT BENCH_REPO N_RUNS LABEL ORDERS START_ITERS PERIODS EPS_MONOTONES VALIDATE_CANDIDATES MAX_RESTARTS DAMPINGS RIDGES DAAREM_ALPHAS DAAREM_KAPPAS CYCL_MONOTONE_TOLS LL_PROBES [N_COMPONENTS]"
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
BENCH_REPO="$2"
N_RUNS="$3"
LABEL="$4"
ORDERS="$5"
START_ITERS="$6"
PERIODS="$7"
EPS_MONOTONES="$8"
VALIDATE_CANDIDATES="$9"
shift 9
MAX_RESTARTS="$1"
DAMPINGS="$2"
RIDGES="$3"
DAAREM_ALPHAS="$4"
DAAREM_KAPPAS="$5"
CYCL_MONOTONE_TOLS="$6"
LL_PROBES="$7"
N_COMPONENTS="${8:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$THREADS}"

mkdir -p "${RUN_DIR}"

CMD=(
  "${PYTHON_BIN}" "${BENCH_REPO}/scripts/run_mica_daarem_sweep.py"
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
  --n-runs "${N_RUNS}"
  --label "${LABEL}"
  --accelerator-orders "${ORDERS}"
  --accelerator-start-iters "${START_ITERS}"
  --accelerator-periods "${PERIODS}"
  --accelerator-eps-monotones "${EPS_MONOTONES}"
  --accelerator-validate-candidates "${VALIDATE_CANDIDATES}"
  --accelerator-max-restarts "${MAX_RESTARTS}"
  --accelerator-dampings "${DAMPINGS}"
  --accelerator-ridges "${RIDGES}"
  --accelerator-daarem-alphas "${DAAREM_ALPHAS}"
  --accelerator-daarem-kappas "${DAAREM_KAPPAS}"
  --accelerator-cycl-monotone-tols "${CYCL_MONOTONE_TOLS}"
  --ll-probes "${LL_PROBES}"
)
if [ -n "${N_COMPONENTS}" ]; then
  CMD+=(--n-components "${N_COMPONENTS}")
fi

"${CMD[@]}"
