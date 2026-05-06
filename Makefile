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
MATLAB_PLOTRESULTS_OUT ?= results/matlab_plotresults_original

MATLAB_DATASET ?= 1
MATLAB_ALGONUM ?= 43

.PHONY: fortran-all python-all fastica-picard-all slurm-fortran slurm-python matlab-image matlab-mutualinfo download-eeglab11 prepare-mica-amica-python matlab-dipfit-smoke matlab-dipfit-amica-python matlab-plotresults-original summarize-mir-no-gv84 figure4b-amica-python

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

# TODO: make command for summarize_benchmark_runs?

download-eeglab11:
	$(PYTHON) scripts/artifacts/download_eeglab11.py --dest $(EEGLAB_DIR)

prepare-mica-amica-python:
	$(PYTHON) scripts/artifacts/prepare_mica_release_for_amica_python.py \
	  --mica-root $(MICA_RELEASE_DIR) \
	  --python-run-root $(PYTHON_RUN_ROOT) \
	  --workdir $(MICA_PREPARED_DIR) \
	  --eeglab-dir $(EEGLAB_DIR) \
	  --force

matlab-mutualinfo:
	$(MATLAB_BIN) -batch "addpath('$(abspath $(EEGLAB_DIR))'); addpath('$(abspath $(MICA_PREPARED_DIR))'); cd('$(abspath $(MICA_PREPARED_DIR))'); mutualinfoalgo"

matlab-dipfit-smoke:
	$(MATLAB_BIN) -batch "addpath('$(abspath $(EEGLAB_DIR))'); addpath('$(abspath $(MICA_PREPARED_DIR))'); cd('$(abspath $(MICA_PREPARED_DIR))'); DATASET=$(MATLAB_DATASET); ALGONUM=$(MATLAB_ALGONUM); processdat"

matlab-dipfit-amica-python:
	$(MATLAB_BIN) -batch "addpath('$(abspath $(EEGLAB_DIR))'); addpath('$(abspath $(MICA_PREPARED_DIR))'); cd('$(abspath $(MICA_PREPARED_DIR))'); run_amica_python_dipfit"

matlab-plotresults-original:
	$(MATLAB_BIN) -batch "addpath('$(abspath $(EEGLAB_DIR))'); addpath('$(abspath $(MICA_PREPARED_DIR))'); cd('$(abspath $(MICA_PREPARED_DIR))'); set(0,'DefaultFigureVisible','off'); plotresults; figs=findall(0,'Type','figure'); outdir='$(abspath $(MATLAB_PLOTRESULTS_OUT))'; if ~exist(outdir,'dir'), mkdir(outdir); end; for k=1:numel(figs), f=figs(k); figure(f); print(f, fullfile(outdir, sprintf('plotresults_fig_%02d.png', k)), '-dpng', '-r300'); print(f, fullfile(outdir, sprintf('plotresults_fig_%02d.pdf', k)), '-dpdf', '-painters'); end; fprintf('Saved %d figures to %s\n', numel(figs), outdir);"

summarize-mir-no-gv84:
	$(PYTHON) scripts/artifacts/summarize_mir_excluding_gv84.py \
	  --mir-mat $(MICA_PREPARED_DIR)/mir_new.mat \
	  --out-prefix results/mir_summary_excluding_gv84

figure4b-amica-python:
	$(PYTHON) scripts/artifacts/plot_delorme_figure4b_with_amica_python.py \
	  --workdir $(MICA_PREPARED_DIR) \
	  --out-prefix results/delorme_figure4b_with_amica_python
