#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from run_mica_fortran import DEFAULT_BENCH_ROOT, DEFAULT_DATASETS_DIR
from run_mica_python import OPTIMIZER_CHOICES, resolve_fortran_out, run_python


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AMICA-Python across all matching mica_release .set datasets with Fortran-initialized weights."
    )
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--dataset-glob", default="*.set")
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--run-tag", default="mica_release_python_all")
    parser.add_argument(
        "--fortran-search-root",
        type=Path,
        default=DEFAULT_BENCH_ROOT,
        help="Root where prior Fortran runs exist (searched for per-dataset fortran_out).",
    )

    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--n-mixtures", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--optimizer", choices=OPTIMIZER_CHOICES, default="em")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--python-threads", type=int, default=4)
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

        f_out = resolve_fortran_out(
            dataset_set=ds,
            fortran_out=None,
            fortran_search_root=args.fortran_search_root,
        )
        manifests.append(
            run_python(
                dataset_set=ds,
                fortran_out=f_out,
                run_dir=ds_run_dir,
                max_iter=args.max_iter,
                n_components=args.n_components,
                n_mixtures=args.n_mixtures,
                random_state=args.random_state,
                device=args.device,
                verbose=args.verbose,
                python_threads=args.python_threads,
                optimizer=args.optimizer,
            )
        )

    summary = {
        "batch_dir": str(batch_dir),
        "fortran_search_root": str(args.fortran_search_root.expanduser().resolve()),
        "optimizer": args.optimizer,
        "n_datasets": len(manifests),
        "datasets": [str(ds) for ds in datasets],
        "runs": manifests,
    }
    out = batch_dir / "python_all_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
