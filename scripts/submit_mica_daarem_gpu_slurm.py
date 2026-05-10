#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

DEFAULT_BENCH_ROOT = Path(__file__).resolve().parents[1] / "benchmark_runs"
DEFAULT_DATASETS_DIR = Path.home() / "amica_test_data" / "mica_release" / "datasets"
DEFAULT_SBATCH = Path(__file__).resolve().parents[1] / "slurm" / "mica_daarem_gpu_dataset.sbatch"
DEFAULT_BENCH_REPO = Path(__file__).resolve().parents[1]
DEFAULT_PARTITION = "gpu"
OMIT_SBATCH_VALUE = {"", "none", "null", "false", "0"}


def add_optional_sbatch_flag(cmd: list[str], flag: str, value: str | None) -> None:
    if value is None:
        return
    normalized = str(value).strip()
    if normalized.lower() in OMIT_SBATCH_VALUE:
        return
    cmd.append(f"{flag}={normalized}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit one GPU DAAREM hyperparameter sweep job per mica_release dataset."
    )
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--dataset-glob", default="*.set")
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--fortran-search-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--run-tag", default="mica_release_daarem_gpu_sweep")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--n-mixtures", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--label", default="")
    parser.add_argument("--accelerator-orders", default="1,2,3")
    parser.add_argument("--accelerator-start-iters", default="1,20")
    parser.add_argument("--accelerator-periods", default="1,3,5")
    parser.add_argument("--accelerator-eps-monotones", default="0")
    parser.add_argument("--accelerator-validate-candidates", default="true")
    parser.add_argument("--accelerator-max-restarts", default="20")
    parser.add_argument("--accelerator-dampings", default="1.0")
    parser.add_argument("--accelerator-ridges", default="1e-8")
    parser.add_argument("--accelerator-daarem-alphas", default="1.2")
    parser.add_argument("--accelerator-daarem-kappas", default="25")
    parser.add_argument("--accelerator-cycl-monotone-tols", default="0")
    parser.add_argument("--ll-probes", default="25,50,100,200")
    parser.add_argument("--partition", default=DEFAULT_PARTITION, help="SLURM partition/queue. Use 'none' to omit.")
    parser.add_argument("--constraint", default=None, help="SLURM node constraint. Use 'none' to omit.")
    parser.add_argument("--gres", default="gpu:1", help="SLURM generic resource request. Use 'none' to omit.")
    parser.add_argument("--time", default="4:00:00")
    parser.add_argument("--mem", default="24G")
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

    label = args.label or f"{args.run_tag}_{ts}"
    for ds in datasets:
        run_dir = batch_dir / ds.stem
        cmd = ["sbatch"]
        add_optional_sbatch_flag(cmd, "--partition", args.partition)
        add_optional_sbatch_flag(cmd, "--constraint", args.constraint)
        add_optional_sbatch_flag(cmd, "--gres", args.gres)
        cmd.extend(
            [
                f"--cpus-per-task={int(args.threads)}",
                f"--mem={args.mem}",
                f"--time={args.time}",
                f"--job-name=micaDG_{ds.stem}",
                str(args.sbatch_script.expanduser().resolve()),
                str(ds.resolve()),
                str(run_dir),
                str(int(args.max_iter)),
                str(int(args.threads)),
                str(args.fortran_search_root.expanduser().resolve()),
                str(int(args.n_mixtures)),
                str(int(args.random_state)),
                str(args.device),
                str(int(args.verbose)),
                str(args.bench_root.expanduser().resolve()),
                str(args.bench_repo.expanduser().resolve()),
                str(int(args.n_runs)),
                label,
                args.accelerator_orders,
                args.accelerator_start_iters,
                args.accelerator_periods,
                args.accelerator_eps_monotones,
                args.accelerator_validate_candidates,
                args.accelerator_max_restarts,
                args.accelerator_dampings,
                args.accelerator_ridges,
                args.accelerator_daarem_alphas,
                args.accelerator_daarem_kappas,
                args.accelerator_cycl_monotone_tols,
                args.ll_probes,
            ]
        )
        if args.n_components is not None:
            cmd.append(str(int(args.n_components)))

        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    print(f"Submitted {len(datasets)} GPU DAAREM sweep jobs. Batch dir: {batch_dir}")


if __name__ == "__main__":
    main()
