# AMICA Benchmark

Benchmark for comparing AMICA-Python against the Fortran AMICA Docker reference on open EEG data.

## To run the benchmark.

Because the benchmark requires both computationally expensive steps (running ICA on
an EEG dataset), and a varied stack (Python for ICA, and MATLAB for replicating results
from a previous paper), this benchmark cannot be simply run with one CLI command. You
will need to follow the steps below. Having access to an HPC would be advantageous if you
wish to re-run ICA on the EEG dataset.

If you don't wish to rerun the benchmark (i.e. ICA on the dataset), you can skip to
steps 5 or 7.

### Instructions:

Part of this process was run on an HPC (for the heavy dute computation), and the rest
was done locally. Initially, I setup my environment on an HPC, the lockfile below
contains the specifcation for that environment:

1. Clone the repo Create and activate the conda environment.

If you haven't already cloned the repo:

```
git clone https://github.com/scott-huberty/amica-benchmark.git
```

```bash
cd amica-benchmark # if needed
conda env create -f environment-linux.yaml
conda activate amica-benchmark
```

2. Download the benchmark dataset and retrieve the Fortran AMICA container.

```bash
python scripts/download_dataset.py

module load apptainer # If needed on your HPC
./scripts/pull_fortran_image.sh shuberty/amica:latest ./amica.sif
```

3. Submit the benchmark jobs.

The paper benchmark used a SLURM scheduler to submit one job per recording.
Each job ran Fortran AMICA, AMICA-Python, and AMICA-Python (DAAREM acceleration) serially
inside the same allocation, in order to ensure as fair a comparison as possible.

I ran the command below 3 times, each time appending a different number (1,2,3) to the
run tag:

```bash
python scripts/submit_mica_triplet_slurm.py \
  --datasets-dir ~/amica_test_data/mica_release/datasets \
  --dataset-glob '*.set' \
  --bench-root ./benchmark_runs \
  --run-tag mica_release_all_run-1 \
  --threads 1 \
  --partition epyc-64 \
  --constraint epyc-7513 \
  --max-iter 2000 \
  --container-runtime apptainer \
  --apptainer-image ./amica.sif \
  --time 12:00:00 \
  --mem 24G
```

Within each dataset directory, outputs are written to `fortran/`, `python_em/`,
and `python_daarem/`.

Note that I constrained the jobs to a specific architecture (`epyc-64`), which is specific
to USC's HPC.

4. Summarize the completed benchmark.

Again, I ran the entire benchmark 3 separate times, and those results were averaged when
reporting runtimes for the paper. I did this to reduce the risk that a single job was unusually
slow due to factors outside my control (e.g. heavy use of the cluster at that moment).

Use `--triplet-batch-glob` to aggregate the benchmark across runs, which is what I did
to summarize the 3 completed runs of each implementation:

```bash
python scripts/summarize_benchmark_runs.py \
  --triplet-batch-glob './benchmark_runs/mica_release_all_run-?_*' \
  --output-dir ./results/benchmark_summary
```

5. Prepare copies of the scripts provided by Delorme 2012 (MIR/DIPFIT calculations).

At this point I moved my working environment to my local computer, because I did not
have MATLAB on my HPC. Again, I created my environment. You can use the same
environment yaml as step 1, or if you are on a Mac or Windows, you can try the crossplatform
environment file, like below:

```bash
cd amica-benchmark
conda env create -f environment.yaml
conda activate amica-benchmark
```

Clone a copy of this repository if needed:

```
git clone https://github.com/scott-huberty/amica-benchmark.git
```

Download the benchmark dataset again if you switch from HPC to a local setup, as the
scripts below will need them:

```bash
python scripts/download_dataset.py
```

And finally, if you ran the benchmark on an HPC, you should use a command like `rsync`,
or `scp` to copy the 3 runs of the benchmark inside `amica-benchark/benchmark_runs/` on
the HPC to the same location in your local copy of the `amica-benchmark` repository.

The first script below will download a specific version of EEGLAB, a matlab toolbox, to
your computer. It is needed to run some of the steps below.

The second script below prepares a working copy of the scripts provided by Delorme et al., 2012.
It patches that copy to add `Py-EM` as algorithm 48 and `Py-DAAREM` as algorithm
49, exports saved AMICA-Python fits (from the joblib files saved by the benchmark) into
`icadecompositions/*.mat`, runs Delorme's original `mutualinfoalgo.m`, and summarizes the
Mutual Information Reduction.

```bash
python scripts/artifacts/download_eeglab11.py \
  --dest ./matlab/eeglab11_0_3_1b

python scripts/artifacts/prepare_mica_release_for_amica_python.py \
  --mica-root ~/amica_test_data/mica_release \
  --triplet-run-dir ./benchmark_runs/mica_release_all_run-1_20260703_115448 \
  --workdir ./benchmark_runs/mica_release_amica_python_matlab \
  --eeglab-dir ./matlab/eeglab11_0_3_1b \
  --force
```

Note that for this portion of the benchmark, we simply chose the 1st run of the benchmark,
as we only were concerned about averaging multiple runs of the benchmark when reporting
runtime, which can vary depending on the load on the node being used.

The command above outputs the artifacts to `benchmark_runs/mica_release_amica_python_matlab`.

6. Calculate MIR and DIPFIT, then summarize the MIR table.

You need to have MATLAB installed on your computer to run steps 6 and 7. I specifically
used version 2023b:

```bash
MATLAB_BIN=/Applications/MATLAB_R2023b.app/bin/matlab # <<< Pass the path to your actual MATLAB executable
EEGLAB_DIR="$(pwd)/matlab/eeglab11_0_3_1b"
MICA_PREPARED_DIR="$(pwd)/benchmark_runs/mica_release_amica_python_matlab"
"$MATLAB_BIN" -batch "addpath('$EEGLAB_DIR'); addpath('$MICA_PREPARED_DIR'); cd('$MICA_PREPARED_DIR'); mutualinfoalgo"

"$MATLAB_BIN" -batch "addpath('$EEGLAB_DIR'); addpath('$MICA_PREPARED_DIR'); cd('$MICA_PREPARED_DIR'); run_amica_python_dipfit"

python scripts/artifacts/summarize_mir_excluding_gv84.py \
  --mir-mat ./benchmark_runs/mica_release_amica_python_matlab/mir_new.mat \
  --out-prefix results/mir_summary_excluding_gv84
```

The MIR summary writes:

```text
results/mir_summary_excluding_gv84.csv
results/mir_summary_excluding_gv84.md
```

The DIPFIT batch writes:

```text
benchmark_runs/mica_release_amica_python_matlab/amica_python_dipfit_batch_results.mat
```

7. Recreate Delorme Figure 4B with the AMICA-Python points.

After both the MATLAB MIR and AMICA-Python DIPFIT commands have completed in
the same prepared working copy, recreate Delorme Figure 4B:

```bash
python scripts/artifacts/plot_delorme_figure4b_with_amica_python.py \
  --workdir ./benchmark_runs/mica_release_amica_python_matlab \
  --out-prefix results/delorme_figure4b_with_amica_python
```

Outputs:

```text
results/delorme_figure4b_with_amica_python.csv
results/delorme_figure4b_with_amica_python.md
results/delorme_figure4b_with_amica_python.png
results/delorme_figure4b_with_amica_python.pdf
```

Voila!
