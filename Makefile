PYTHON ?= python
DATASETS_DIR ?= $(HOME)/amica_test_data/mica_release/datasets
BENCH_ROOT ?= ./benchmark_runs
FORTRAN_IMAGE ?= shuberty/amica:latest
CONTAINER_RUNTIME ?= docker
APPTAINER_IMAGE ?=
THREADS ?= 4
MAX_ITER ?= 2000
MATLAB_IMAGE ?= mica-matlab:r2012b
MATLAB_LICENSE_SERVER ?=
MATLAB_FILE_INSTALLATION_KEY ?=
MICA_RELEASE_DIR ?= $(HOME)/amica_test_data/mica_release
MICA_PREPARED_DIR ?= $(BENCH_ROOT)/mica_release_amica_python_matlab
EEGLAB_DIR ?= ./matlab/eeglab11_0_3_1b
MATLAB_BIN ?= /Applications/MATLAB_R2023b.app/bin/matlab
PYTHON_RUN_ROOT ?= $(BENCH_ROOT)/mica_release_python_slurm_20260419_174859

.PHONY: fortran-all python-all fastica-picard-all slurm-fortran slurm-python matlab-image matlab-mutualinfo download-eeglab11 prepare-mica-amica-python matlab-mutualinfo-local summarize-mir-no-gv84

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

fastica-picard-all:
	$(PYTHON) scripts/run_mica_fastica_picard_all.py \
	  --datasets-dir $(DATASETS_DIR) \
	  --bench-root $(BENCH_ROOT) \
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

download-eeglab11:
	$(PYTHON) scripts/artifacts/download_eeglab11.py --dest $(EEGLAB_DIR)

prepare-mica-amica-python:
	$(PYTHON) scripts/artifacts/prepare_mica_release_for_amica_python.py \
	  --mica-root $(MICA_RELEASE_DIR) \
	  --python-run-root $(PYTHON_RUN_ROOT) \
	  --workdir $(MICA_PREPARED_DIR) \
	  --eeglab-dir $(EEGLAB_DIR) \
	  --force

matlab-mutualinfo-local:
	$(MATLAB_BIN) -batch "addpath('$(abspath $(EEGLAB_DIR))'); addpath('$(abspath $(MICA_PREPARED_DIR))'); cd('$(abspath $(MICA_PREPARED_DIR))'); mutualinfoalgo"

summarize-mir-no-gv84:
	$(PYTHON) scripts/artifacts/summarize_mir_excluding_gv84.py \
	  --mir-mat $(MICA_PREPARED_DIR)/mir_new.mat \
	  --out-prefix results/mir_summary_excluding_gv84
