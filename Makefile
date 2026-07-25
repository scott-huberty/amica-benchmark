PYTHON ?= python
DATASETS_DIR ?= $(HOME)/amica_test_data/mica_release/datasets
BENCH_ROOT ?= ./benchmark_runs
FORTRAN_IMAGE ?= shuberty/amica:latest
CONTAINER_RUNTIME ?= apptainer
APPTAINER_IMAGE ?= ./amica.sif
THREADS ?= 1
MAX_ITER ?= 2000
SLURM_PARTITION ?= epyc-64
SLURM_CONSTRAINT ?= epyc-7513
SLURM_TIME ?= 12:00:00
SLURM_MEM ?= 24G
RUN_TAG ?= mica_release_all_run-1
TRIPLET_BATCH_DIR ?= $(BENCH_ROOT)/mica_release_triplet_golden_20260511_232913
TRIPLET_BATCH_GLOB ?= $(BENCH_ROOT)/mica_release_all_run-?_*
BENCHMARK_SUMMARY_DIR ?= results/benchmark_summary
MICA_RELEASE_DIR ?= $(HOME)/amica_test_data/mica_release
TRIPLET_RUN_DIR ?= $(BENCH_ROOT)/mica_release_all_run-1_20260703_115448
MICA_PREPARED_DIR ?= $(BENCH_ROOT)/mica_release_amica_python_matlab
EEGLAB_DIR ?= ./matlab/eeglab11_0_3_1b
MATLAB_BIN ?= /Applications/MATLAB_R2023b.app/bin/matlab
MIR_OUT_PREFIX ?= results/mir_summary_excluding_gv84
FIGURE4B_OUT_PREFIX ?= results/delorme_figure4b_with_amica_python
MATLAB_DATASET ?= 1
MATLAB_ALGONUM ?= 43

.PHONY: help download-dataset pull-fortran-image submit-triplet summarize-triplet \
	summarize-triplet-runs download-eeglab11 prepare-mica-amica-python \
	matlab-mutualinfo matlab-dipfit-amica-python summarize-mir-no-gv84 \
	figure4b-amica-python matlab-dipfit-smoke fortran-all python-all \
	fastica-picard-all slurm-fortran slurm-python

help:
	@printf '%s\n' \
	  'Makefile shortcuts assume the conda environment is already activated.' \
	  '' \
	  'Paper workflow targets:' \
	  '  make download-dataset' \
	  '  make pull-fortran-image CONTAINER_RUNTIME=apptainer APPTAINER_IMAGE=./amica.sif' \
	  '  make submit-triplet RUN_TAG=mica_release_all_run-1' \
	  '  make summarize-triplet-runs' \
	  '  make download-eeglab11 prepare-mica-amica-python' \
	  '  make matlab-mutualinfo matlab-dipfit-amica-python summarize-mir-no-gv84' \
	  '  make figure4b-amica-python' \
	  '' \
	  'Useful overrides: DATASETS_DIR, BENCH_ROOT, THREADS, MAX_ITER,' \
	  '  SLURM_PARTITION, SLURM_CONSTRAINT, MATLAB_BIN, TRIPLET_RUN_DIR.'

download-dataset:
	$(PYTHON) scripts/download_dataset.py

pull-fortran-image:
	RUNTIME=$(CONTAINER_RUNTIME) ./scripts/pull_fortran_image.sh $(FORTRAN_IMAGE) $(APPTAINER_IMAGE)

submit-triplet:
	$(PYTHON) scripts/submit_mica_triplet_slurm.py \
	  --datasets-dir $(DATASETS_DIR) \
	  --dataset-glob '*.set' \
	  --bench-root $(BENCH_ROOT) \
	  --run-tag $(RUN_TAG) \
	  --threads $(THREADS) \
	  $(if $(SLURM_PARTITION),--partition $(SLURM_PARTITION),--partition none) \
	  $(if $(SLURM_CONSTRAINT),--constraint $(SLURM_CONSTRAINT),--constraint none) \
	  --max-iter $(MAX_ITER) \
	  --container-runtime $(CONTAINER_RUNTIME) \
	  $(if $(APPTAINER_IMAGE),--apptainer-image $(APPTAINER_IMAGE),) \
	  --time $(SLURM_TIME) \
	  --mem $(SLURM_MEM)

summarize-triplet:
	$(PYTHON) scripts/summarize_benchmark_runs.py \
	  --triplet-batch-dir $(TRIPLET_BATCH_DIR) \
	  --output-dir $(BENCHMARK_SUMMARY_DIR)

summarize-triplet-runs:
	$(PYTHON) scripts/summarize_benchmark_runs.py \
	  --triplet-batch-glob '$(TRIPLET_BATCH_GLOB)' \
	  --output-dir $(BENCHMARK_SUMMARY_DIR)

download-eeglab11:
	$(PYTHON) scripts/artifacts/download_eeglab11.py \
	  --dest $(EEGLAB_DIR)

prepare-mica-amica-python:
	$(PYTHON) scripts/artifacts/prepare_mica_release_for_amica_python.py \
	  --mica-root $(MICA_RELEASE_DIR) \
	  --triplet-run-dir $(TRIPLET_RUN_DIR) \
	  --workdir $(MICA_PREPARED_DIR) \
	  --eeglab-dir $(EEGLAB_DIR) \
	  --force

matlab-mutualinfo:
	$(MATLAB_BIN) -batch "addpath('$(abspath $(EEGLAB_DIR))'); addpath('$(abspath $(MICA_PREPARED_DIR))'); cd('$(abspath $(MICA_PREPARED_DIR))'); mutualinfoalgo"

matlab-dipfit-amica-python:
	$(MATLAB_BIN) -batch "addpath('$(abspath $(EEGLAB_DIR))'); addpath('$(abspath $(MICA_PREPARED_DIR))'); cd('$(abspath $(MICA_PREPARED_DIR))'); run_amica_python_dipfit"

summarize-mir-no-gv84:
	$(PYTHON) scripts/artifacts/summarize_mir_excluding_gv84.py \
	  --mir-mat $(MICA_PREPARED_DIR)/mir_new.mat \
	  --out-prefix $(MIR_OUT_PREFIX)

figure4b-amica-python:
	$(PYTHON) scripts/artifacts/plot_delorme_figure4b_with_amica_python.py \
	  --workdir $(MICA_PREPARED_DIR) \
	  --out-prefix $(FIGURE4B_OUT_PREFIX)

matlab-dipfit-smoke:
	$(MATLAB_BIN) -batch "addpath('$(abspath $(EEGLAB_DIR))'); addpath('$(abspath $(MICA_PREPARED_DIR))'); cd('$(abspath $(MICA_PREPARED_DIR))'); DATASET=$(MATLAB_DATASET); ALGONUM=$(MATLAB_ALGONUM); processdat"

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
	  $(if $(SLURM_PARTITION),--partition $(SLURM_PARTITION),--partition none) \
	  $(if $(SLURM_CONSTRAINT),--constraint $(SLURM_CONSTRAINT),--constraint none) \
	  --max-iter $(MAX_ITER)

slurm-python:
	$(PYTHON) scripts/submit_mica_python_slurm.py \
	  --datasets-dir $(DATASETS_DIR) \
	  --bench-root $(BENCH_ROOT) \
	  --fortran-search-root $(BENCH_ROOT) \
	  --threads $(THREADS) \
	  $(if $(SLURM_PARTITION),--partition $(SLURM_PARTITION),--partition none) \
	  $(if $(SLURM_CONSTRAINT),--constraint $(SLURM_CONSTRAINT),--constraint none) \
	  --max-iter $(MAX_ITER)
