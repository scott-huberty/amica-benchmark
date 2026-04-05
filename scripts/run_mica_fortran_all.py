#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from run_mica_fortran import DEFAULT_BENCH_ROOT, DEFAULT_DATASETS_DIR, run_fortran


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fortran AMICA across all matching mica_release .set datasets.")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--dataset-glob", default="*.set")
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--run-tag", default="mica_release_fortran_all")

    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--n-mixtures", type=int, default=3)
    parser.add_argument(
        "--container-runtime",
        choices=("docker", "apptainer"),
        default="docker",
    )
    parser.add_argument("--fortran-image", default="shuberty/amica:latest")
    parser.add_argument("--apptainer-image", default=None)
    parser.add_argument("--fortran-threads", type=int, default=4)
    args = parser.parse_args()

    datasets = sorted(args.datasets_dir.expanduser().resolve().glob(args.dataset_glob))
    if not datasets:
        raise FileNotFoundError(f"No datasets found: {args.datasets_dir} glob={args.dataset_glob}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = args.bench_root.expanduser().resolve() / f"{args.run_tag}_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, object]] = []
    for ds in datasets:
        ds_run_dir = batch_dir / ds.stem
        ds_run_dir.mkdir(parents=True, exist_ok=True)
        manifests.append(
            run_fortran(
                dataset_set=ds,
                run_dir=ds_run_dir,
                max_iter=args.max_iter,
                n_components=args.n_components,
                n_mixtures=args.n_mixtures,
                fortran_image=args.fortran_image,
                container_runtime=args.container_runtime,
                apptainer_image=args.apptainer_image,
                fortran_threads=args.fortran_threads,
            )
        )

    summary = {
        "batch_dir": str(batch_dir),
        "n_datasets": len(manifests),
        "datasets": [str(ds) for ds in datasets],
        "runs": manifests,
    }
    out = batch_dir / "fortran_all_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
