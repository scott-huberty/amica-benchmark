PYTHON ?= python
DATASETS_DIR ?= $(HOME)/amica_test_data/mica_release/datasets
BENCH_ROOT ?= ./benchmark_runs
FORTRAN_IMAGE ?= shuberty/amica:latest
CONTAINER_RUNTIME ?= docker
APPTAINER_IMAGE ?=
THREADS ?= 4
MAX_ITER ?= 2000

.PHONY: fortran-all python-all slurm-fortran slurm-python

fortran-all:
	$(PYTHON) scripts/run_mica_fortran_all.py \
	  --datasets-dir $(DATASETS_DIR) \
	  --bench-root $(BENCH_ROOT) \
	  --container-runtime $(CONTAINER_RUNTIME) \
	  --fortran-image $(FORTRAN_IMAGE) \
	  $(if $(APPTAINER_IMAGE),--apptainer-image $(APPTAINER_IMAGE),) \
	  --fortran-threads $(THREADS) \
	  --max-iter $(MAX_ITER)

python-all:
	$(PYTHON) scripts/run_mica_python_all.py \
	  --datasets-dir $(DATASETS_DIR) \
	  --bench-root $(BENCH_ROOT) \
	  --fortran-search-root $(BENCH_ROOT) \
	  --python-threads $(THREADS) \
	  --max-iter $(MAX_ITER)

slurm-fortran:
	$(PYTHON) scripts/submit_mica_fortran_slurm.py \
	  --datasets-dir $(DATASETS_DIR) \
	  --bench-root $(BENCH_ROOT) \
	  --container-runtime $(CONTAINER_RUNTIME) \
	  --fortran-image $(FORTRAN_IMAGE) \
	  $(if $(APPTAINER_IMAGE),--apptainer-image $(APPTAINER_IMAGE),) \
	  --threads $(THREADS) \
	  --max-iter $(MAX_ITER)

slurm-python:
	$(PYTHON) scripts/submit_mica_python_slurm.py \
	  --datasets-dir $(DATASETS_DIR) \
	  --bench-root $(BENCH_ROOT) \
	  --fortran-search-root $(BENCH_ROOT) \
	  --threads $(THREADS) \
	  --max-iter $(MAX_ITER)
