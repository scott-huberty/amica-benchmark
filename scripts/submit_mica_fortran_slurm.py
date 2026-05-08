#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

DEFAULT_BENCH_ROOT = Path(__file__).resolve().parents[1] / "benchmark_runs"
DEFAULT_DATASETS_DIR = Path.home() / "amica_test_data" / "mica_release" / "datasets"
DEFAULT_SBATCH = Path(__file__).resolve().parents[1] / "slurm" / "mica_fortran_dataset.sbatch"
DEFAULT_BENCH_REPO = Path(__file__).resolve().parents[1]
DEFAULT_PARTITION = "epyc-64"
DEFAULT_CONSTRAINT = "epyc-7513"
OMIT_SBATCH_VALUE = {"", "none", "null", "false", "0"}


def add_optional_sbatch_flag(cmd: list[str], flag: str, value: str | None) -> None:
    if value is None:
        return
    normalized = str(value).strip()
    if normalized.lower() in OMIT_SBATCH_VALUE:
        return
    cmd.append(f"{flag}={normalized}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit one Fortran SLURM job per dataset.")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--dataset-glob", default="*.set")
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--run-tag", default="mica_release_fortran_slurm")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--n-mixtures", type=int, default=3)
    parser.add_argument(
        "--container-runtime",
        choices=("docker", "apptainer"),
        default="apptainer",
    )
    parser.add_argument("--fortran-image", default="shuberty/amica:latest")
    parser.add_argument("--apptainer-image", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--partition", default=DEFAULT_PARTITION, help="SLURM partition/queue. Use 'none' to omit.")
    parser.add_argument("--constraint", default=DEFAULT_CONSTRAINT, help="SLURM node constraint. Use 'none' to omit.")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--bench-repo", type=Path, default=DEFAULT_BENCH_REPO)
    parser.add_argument("--sbatch-script", type=Path, default=DEFAULT_SBATCH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = sorted(args.datasets_dir.expanduser().resolve().glob(args.dataset_glob))
    if not datasets:
        raise FileNotFoundError(f"No datasets found: {args.datasets_dir} glob={args.dataset_glob}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = args.bench_root.expanduser().resolve() / f"{args.run_tag}_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        run_dir = batch_dir / ds.stem
        cmd = ["sbatch"]
        add_optional_sbatch_flag(cmd, "--partition", args.partition)
        add_optional_sbatch_flag(cmd, "--constraint", args.constraint)
        cmd.extend(
            [
                f"--cpus-per-task={int(args.threads)}",
                f"--mem={args.mem}",
                f"--time={args.time}",
                f"--job-name=micaF_{ds.stem}",
                str(args.sbatch_script.expanduser().resolve()),
                str(ds.resolve()),
                str(run_dir),
                str(int(args.max_iter)),
                str(int(args.threads)),
                str(args.fortran_image),
                str(int(args.n_mixtures)),
                str(args.container_runtime),
                "" if args.apptainer_image is None else str(args.apptainer_image),
                str(args.bench_repo.expanduser().resolve()),
            ]
        )
        if args.n_components is not None:
            cmd.append(str(int(args.n_components)))

        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    print(f"Submitted {len(datasets)} Fortran jobs. Batch dir: {batch_dir}")


if __name__ == "__main__":
    main()
