# AMICA Benchmark

Benchmark harness for comparing AMICA-Python against the Fortran AMICA Docker reference on open EEG data.

## Setup

Choose one environment manager:

```bash
cd /Users/scotterik/devel/projects/amica-python/amica-benchmark
pixi install
```

```bash
cd /Users/scotterik/devel/projects/amica-python/amica-benchmark
uv sync
```

```bash
cd /Users/scotterik/devel/projects/amica-python/amica-benchmark
conda env create -f environment.yaml
conda activate amica-benchmark
```

## Common workflows

Use whichever launcher matches your environment:

```bash
pixi run download-dataset
uv run python scripts/download_dataset.py
```

```bash
pixi run benchmark-fortran-all
uv run python scripts/run_mica_fortran_all.py
```

```bash
pixi run benchmark-python-all
uv run python scripts/run_mica_python_all.py
```

```bash
pixi run benchmark-fastica-picard-all
uv run python scripts/run_mica_fastica_picard_all.py
```

```bash
pixi run benchmark-slurm-fortran
pixi run benchmark-slurm-python
uv run python scripts/submit_mica_fortran_slurm.py
uv run python scripts/submit_mica_python_slurm.py
```

Pass script options directly when needed:

```bash
pixi run benchmark-python-all -- --datasets-dir ~/amica_test_data/mica_release/datasets --fortran-search-root ./benchmark_runs --python-threads 4 --max-iter 1000
uv run python scripts/run_mica_python_all.py --datasets-dir ~/amica_test_data/mica_release/datasets --fortran-search-root ./benchmark_runs --python-threads 4 --max-iter 1000
```

For single-subject runs, call the scripts directly:

```bash
python scripts/run_mica_fortran.py --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set
python scripts/run_mica_python.py --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set
python scripts/run_mica_fastica_picard.py --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set
```

## Fortran image

```bash
./scripts/pull_fortran_image.sh shuberty/amica:latest
```

For cluster / Apptainer workflows:

```bash
RUNTIME=apptainer ./scripts/pull_fortran_image.sh shuberty/amica:latest /path/to/save/your/containers/amica.sif
```

Note that depending on your HPC you may have to load apptainer via `LMOD` before running the above command, e.g. `module load apptainer`


## Recompute Delorme MIR with AMICA-Python

The reproducible MATLAB workflow prepares a working copy of the MICA release,
patches that copy to add `AMICA-Python` as algorithm 48, exports saved
AMICA-Python joblib fits into `icadecompositions/*.mat`, runs Delorme's
original `mutualinfoalgo.m`, and summarizes MIR while excluding MATLAB dataset
10 (`gv84`) from every algorithm.

EEGLAB 11.0.3.1b is expected at `./matlab/eeglab11_0_3_1b`. If it is missing:

```bash
make download-eeglab11
```

Prepare the MICA working copy and AMICA-Python decomposition files:

```bash
make prepare-mica-amica-python \
  PYTHON=/Users/scotterik/miniforge3/envs/amica_env/bin/python \
  MICA_RELEASE_DIR=/Users/scotterik/amica_test_data/mica_release \
  PYTHON_RUN_ROOT=./benchmark_runs/mica_release_python_slurm_20260419_174859
```


This copies the mica release directoryto `benchmark_runs/mica_release_amica_python_matlab`,
and patches to `mutualinfoalgo` to incluce a mat file of AMICA-Python results.

Run the MATLAB MIR script:

```bash
make matlab-mutualinfo \
  MATLAB_BIN=/Applications/MATLAB_R2023b.app/bin/matlab
```

Summarize the result excluding `gv84`:

```bash
make summarize-mir-no-gv84 \
  PYTHON=/Users/scotterik/miniforge3/envs/amica_env/bin/python
```

Outputs:

```text
results/mir_summary_excluding_gv84.csv
results/mir_summary_excluding_gv84.md
```

To smoke-test the original MATLAB/DIPFIT fitting path on one dataset and
algorithm:

```bash
make matlab-dipfit-smoke \
  MATLAB_BIN=/Applications/MATLAB_R2023b.app/bin/matlab \
  MATLAB_DATASET=1 \
  MATLAB_ALGONUM=43
```

This runs `DATASET=1; ALGONUM=43; processdat` in the prepared working copy.
Per `processdat.m`, MATLAB dataset 10 is `gv84`.

To run DIPFIT for AMICA-Python only, excluding `gv84`:

```bash
make matlab-dipfit-amica-python \
  MATLAB_BIN=/Applications/MATLAB_R2023b.app/bin/matlab
```

This runs `run_amica_python_dipfit.m` in the prepared working copy. It writes
updated AMICA-Python decomposition files for `ALGONUM=48` and saves a batch
status file:

```text
benchmark_runs/mica_release_amica_python_matlab/amica_python_dipfit_batch_results.mat
```

After both `matlab-mutualinfo` and `matlab-dipfit-amica-python` have
completed in the same prepared working copy, recreate Delorme Figure 4B and add
AMICA-Python:

```bash
make figure4b-amica-python \
  PYTHON=/Users/scotterik/miniforge3/envs/amica_env/bin/python
```

Outputs:

```text
results/delorme_figure4b_with_amica_python.csv
results/delorme_figure4b_with_amica_python.md
results/delorme_figure4b_with_amica_python.png
results/delorme_figure4b_with_amica_python.pdf
```

## Single-dataset runners

```bash
python scripts/run_mica_fortran.py \
  --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set \
  --container-runtime docker \
  --max-iter 1000 \
  --fortran-threads 4
```

On clusters, use Apptainer:

```bash
python scripts/run_mica_fortran.py \
  --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set \
  --container-runtime apptainer \
  --apptainer-image /shared/containers/amica.sif \
  --max-iter 1000 \
  --fortran-threads 4
```

```bash
python scripts/run_mica_python.py \
  --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set \
  --fortran-search-root ./benchmark_runs \
  --max-iter 1000 \
  --python-threads 4
```

If `--fortran-out` is omitted, `run_mica_python.py` searches `--fortran-search-root` for a matching `fortran_out` directory by dataset stem, preferring exact stem folder matches and then newest outputs.

Run the simpler Delorme-style comparison algorithms for one subject:

```bash
python scripts/run_mica_fastica_picard.py \
  --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set \
  --max-iter 1000 \
  --python-threads 4
```

This fits scikit-learn FastICA and Picard extended-infomax (`extended=True`, `ortho=False`), saves both fitted estimators as joblib files, records wall-clock fit time, and writes Delorme `getent2`-style mutual-information reduction into `fastica_picard_run.json`.

## All-dataset local runs

If you have a virtual environment activated, you can use the makefile:

```bash
make fortran-all THREADS=4 MAX_ITER=1000
make python-all THREADS=4 MAX_ITER=1000
make fastica-picard-all THREADS=4 MAX_ITER=1000
```

You can also specify a specific python or python binary:

```bash
make PYTHON=python3 python-all
make PYTHON=~/envs/miniforge/amica_env/bin/python python-all
```

## SLURM: one job per dataset

Each submitted job requests:
- `--cpus-per-task=4`
- `--mem=16G`
- `--time=3:00:00`
- `--max-iter=2000`
- `--partition=epyc-64`
- `--constraint=epyc-7513`
- Python submitter verbosity defaults to `--verbose 2`

The partition and constraint defaults are for USC CARC Discovery and are used to
reduce run-to-run performance swings by keeping benchmark jobs on the same CPU
architecture. When running on another HPC, set these to values that exist on that
cluster. If your scheduler does not use node constraints, or you do not want to
pin a CPU model, pass `--constraint none` with the submitters or
`SLURM_CONSTRAINT=none` with `make`; `none`, `null`, `false`, `0`, and an empty
string are treated as "do not pass `--constraint`". You can similarly omit
`--partition` by passing `--partition none`, though most clusters require or
prefer an explicit partition/queue.

Example for another HPC:

```bash
python scripts/submit_mica_python_slurm.py \
  --partition cpu \
  --constraint none \
  --threads 4 \
  --max-iter 2000
```

Equivalent `make` override:

```bash
make slurm-python THREADS=4 MAX_ITER=2000 SLURM_PARTITION=cpu SLURM_CONSTRAINT=none
```

Submit Fortran jobs:

```bash
make slurm-fortran THREADS=4 MAX_ITER=2000
```

For Apptainer-backed SLURM submission:

```bash
make slurm-fortran THREADS=4 MAX_ITER=2000 CONTAINER_RUNTIME=apptainer APPTAINER_IMAGE=/path/to/containers/amica.sif
```

To run just a single subject:

```bash
python scripts/submit_mica_fortran_slurm.py \
  --datasets-dir ~/amica_test_data/mica_release/datasets \
  --dataset-glob 'cz84.set' \
  --partition epyc-64 \
  --constraint epyc-7513 \
  --threads 4 \
  --max-iter 2000 \
  --container-runtime apptainer \
  --apptainer-image /path/to/containers/amica.sif
```
 or for python

 ```bash
 python scripts/submit_mica_python_slurm.py \
  --datasets-dir ~/amica_test_data/mica_release/datasets \
  --dataset-glob 'cz84.set' \
  --fortran-search-root ./benchmark_runs \
  --partition epyc-64 \
  --constraint epyc-7513 \
  --threads 4 \
  --max-iter 2000
```

Submit Python jobs:

```bash
make slurm-python THREADS=4 MAX_ITER=2000
```

To submit the Python batch with the DAAREM optimizer, invoke the submitter
directly and pass `--optimizer daarem`. Use `--accelerator-order` to override
the AMICA-Python default DAAREM history/order for the run:

```bash
/Users/scotterik/miniforge3/envs/amica_env/bin/python scripts/submit_mica_python_slurm.py \
  --datasets-dir ~/amica_test_data/mica_release/datasets \
  --dataset-glob '*.set' \
  --bench-root ./benchmark_runs \
  --fortran-search-root ./benchmark_runs \
  --threads 4 \
  --partition epyc-64 \
  --constraint epyc-7513 \
  --max-iter 2000 \
  --optimizer daarem \
  --accelerator-order 3
```

The default optimizer is `em`, so omit `--optimizer` or pass `--optimizer em`
to run the original AMICA-Python optimization path.

To submit one GPU DAAREM hyperparameter sweep job per MICA recording:

```bash
cd /Users/scotterik/devel/projects/amica-python/amica-benchmark

PYTHON_BIN=/Users/scotterik/miniforge3/envs/amica_env/bin/python \
/Users/scotterik/miniforge3/envs/amica_env/bin/python scripts/submit_mica_daarem_gpu_slurm.py \
  --datasets-dir ~/amica_test_data/mica_release/datasets \
  --dataset-glob '*.set' \
  --bench-root ./benchmark_runs \
  --fortran-search-root ./benchmark_runs/mica_release_fortran_slurm_20260509_230036 \
  --run-tag mica_release_daarem_gpu_sweep \
  --max-iter 2000 \
  --n-runs 1 \
  --partition gpu \
  --gres gpu:1 \
  --threads 2 \
  --mem 24G \
  --time 4:00:00
```

The default sweep covers `accelerator_order=1,2,3`,
`accelerator_start_iter=1,5,10,25`, `accelerator_period=1,2,5,10`, and
`accelerator_validate_candidate=true,false`, for 96 fits per recording. Each
recording writes `daarem_sweep.csv`, `daarem_ll_curves.npz`, and
`daarem_sweep_run.json` under its run directory. Adjust `--partition` and
`--gres` if your cluster uses different GPU resource names.

Or invoke submitters directly:

```bash
python scripts/submit_mica_fortran_slurm.py --threads 4 --max-iter 2000 --partition epyc-64 --constraint epyc-7513
python scripts/submit_mica_python_slurm.py --threads 4 --max-iter 2000 --partition epyc-64 --constraint epyc-7513 --optimizer em
```

Summarize a completed Fortran/Python benchmark pair:

```bash
/Users/scotterik/miniforge3/envs/amica_env/bin/python /Users/scotterik/devel/projects/amica-python/amica-benchmark/scripts/summarize_benchmark_runs.py \
  --fortran-batch-dir /Users/scotterik/devel/projects/amica-python/amica-benchmark/benchmark_runs/mica_release_fortran_slurm_20260419_164625 \
  --python-batch-dir /Users/scotterik/devel/projects/amica-python/amica-benchmark/benchmark_runs/mica_release_python_slurm_20260419_174859
```
