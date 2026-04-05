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
pixi run benchmark-slurm-fortran
pixi run benchmark-slurm-python
uv run python scripts/submit_mica_fortran_slurm.py
uv run python scripts/submit_mica_python_slurm.py
```

Pass script options directly when needed:

```bash
pixi run benchmark-python-all -- --datasets-dir ~/amica_test_data/mica_release/datasets --fortran-search-root ./benchmark_runs --python-threads 4 --max-iter 2000
uv run python scripts/run_mica_python_all.py --datasets-dir ~/amica_test_data/mica_release/datasets --fortran-search-root ./benchmark_runs --python-threads 4 --max-iter 2000
```

For single-subject runs, call the scripts directly:

```bash
python scripts/run_mica_fortran.py --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set
python scripts/run_mica_python.py --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set
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

## Single-dataset runners

```bash
python scripts/run_mica_fortran.py \
  --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set \
  --container-runtime docker \
  --max-iter 2000 \
  --fortran-threads 4
```

On clusters, use Apptainer:

```bash
python scripts/run_mica_fortran.py \
  --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set \
  --container-runtime apptainer \
  --apptainer-image /shared/containers/amica.sif \
  --max-iter 2000 \
  --fortran-threads 4
```

```bash
python scripts/run_mica_python.py \
  --dataset-set ~/amica_test_data/mica_release/datasets/cz84.set \
  --fortran-search-root ./benchmark_runs \
  --max-iter 2000 \
  --python-threads 4
```

If `--fortran-out` is omitted, `run_mica_python.py` searches `--fortran-search-root` for a matching `fortran_out` directory by dataset stem, preferring exact stem folder matches and then newest outputs.

## All-dataset local runs

If you have a virtual environment activated, you can use the makefile:

```bash
make fortran-all THREADS=4 MAX_ITER=2000
make python-all THREADS=4 MAX_ITER=2000
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
- `--time=12:00:00`
- `--max-iter=2000`

Submit Fortran jobs:

```bash
make slurm-fortran THREADS=4 MAX_ITER=2000
```

For Apptainer-backed SLURM submission:

```bash
make slurm-fortran THREADS=4 MAX_ITER=2000 CONTAINER_RUNTIME=apptainer APPTAINER_IMAGE=/path/to/containers/amica.sif
```

To one just a single subject:

```bash
python scripts/submit_mica_fortran_slurm.py \
  --datasets-dir ~/amica_test_data/mica_release/datasets \
  --dataset-glob 'cz84.set' \
  --threads 4 \
  --max-iter 2000 \
  --container-runtime apptainer \
  --apptainer-image /path/to/containers/amica.sif
```

Submit Python jobs:

```bash
make slurm-python THREADS=4 MAX_ITER=2000
```

Or invoke submitters directly:

```bash
python scripts/submit_mica_fortran_slurm.py --threads 4 --max-iter 2000
python scripts/submit_mica_python_slurm.py --threads 4 --max-iter 2000
```
