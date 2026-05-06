#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import psutil

from amica import AMICA
from amica.utils.fortran import load_fortran_results, load_initial_weights

from mica_common import discover_fortran_out, infer_fortran_n_components, load_eeglab_set


DEFAULT_BENCH_ROOT = Path(__file__).resolve().parents[1] / "benchmark_runs"
DEFAULT_DATASETS_DIR = Path.home() / "amica_test_data" / "mica_release" / "datasets"
OPTIMIZER_CHOICES = ("em", "daarem")


def collect_environment_metadata() -> dict[str, object]:
    return {
        "sys_platform": sys.platform,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "pid": os.getpid(),
        "os_cpu_count": os.cpu_count(),
    }


def resolve_fortran_out(
    *,
    dataset_set: Path,
    fortran_out: Path | None,
    fortran_search_root: Path,
) -> Path:
    if fortran_out is not None:
        resolved = fortran_out.expanduser().resolve()
        if not (resolved / "W").exists():
            raise FileNotFoundError(f"Fortran W file not found: {resolved / 'W'}")
        return resolved
    return discover_fortran_out(dataset_set=dataset_set, search_root=fortran_search_root)


def run_python(
    *,
    dataset_set: Path,
    fortran_out: Path,
    run_dir: Path,
    max_iter: int,
    n_components: int | None,
    n_mixtures: int,
    random_state: int,
    device: str,
    verbose: int,
    python_threads: int,
    optimizer: str = "em",
) -> dict[str, object]:
    dataset_set = dataset_set.expanduser().resolve()
    fortran_out = fortran_out.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_eeglab_set(dataset_set)
    n_features = int(data.shape[1])

    if n_components is None:
        n_components = infer_fortran_n_components(fortran_out)

    w_init, sbeta_init, mu_init = load_initial_weights(
        fortran_out,
        n_components=int(n_components),
        n_mixtures=int(n_mixtures),
    )

    os.environ["OPM_NUM_THREADS"] = str(int(python_threads))
    os.environ["MKL_NUM_THREADS"] = str(int(python_threads))

    model = AMICA(
        n_components=None if n_components is None else int(n_components),
        n_mixtures=int(n_mixtures),
        max_iter=int(max_iter),
        random_state=int(random_state),
        device=device,
        verbose=int(verbose),
        optimizer=optimizer,
        w_init=w_init,
        sbeta_init=sbeta_init,
        mu_init=mu_init,
    )
    fit_started_at_epoch_seconds = time.time()
    fit_started_at_perf_counter = time.perf_counter()
    model.fit(data)
    fit_finished_at_epoch_seconds = time.time()
    fit_seconds = time.perf_counter() - fit_started_at_perf_counter

    model_path = run_dir / "python_model.joblib"
    joblib.dump(model, model_path)

    ll = np.asarray(model.ll_ if hasattr(model, "ll_") else model._ll)
    components = np.asarray(model.components_)

    out_npz = run_dir / "python_results.npz"
    np.savez(
        out_npz,
        ll=ll,
        components=components,
        mixing=np.asarray(model.mixing_),
        unmixing=np.asarray(model._unmixing),
        whitening=np.asarray(model.whitening_),
    )

    fres = load_fortran_results(
        fortran_out,
        n_components=int(n_components),
        n_mixtures=int(n_mixtures),
        n_features=n_features,
    )

    ll_f = fres["LL"]
    ll_f_nz = ll_f[ll_f != 0]
    ll_p_nz = ll[ll != 0]

    manifest = {
        "dataset_set": str(dataset_set),
        "fortran_output_dir": str(fortran_out),
        "run_dir": str(run_dir),
        "data_shape_samples_features": [int(data.shape[0]), int(data.shape[1])],
        "max_iter": int(max_iter),
        "n_components": int(n_components),
        "n_mixtures": int(n_mixtures),
        "random_state": int(random_state),
        "device": device,
        "optimizer": optimizer,
        "python_threads": int(python_threads),
        "environment": collect_environment_metadata(),
        "fit_started_at_epoch_seconds": float(fit_started_at_epoch_seconds),
        "fit_finished_at_epoch_seconds": float(fit_finished_at_epoch_seconds),
        "fit_seconds": float(fit_seconds),
        "python_results_npz": str(out_npz),
        "python_model_joblib": str(model_path),
        "fortran_ll_final": float(ll_f_nz[-1]) if ll_f_nz.size else float("nan"),
        "python_ll_final": float(ll_p_nz[-1]) if ll_p_nz.size else float("nan"),
    }
    manifest_path = run_dir / "python_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _default_run_dir(dataset_set: Path, bench_root: Path) -> Path:
    return (bench_root / f"mica_release_{dataset_set.stem}_python").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AMICA-Python on one EEGLAB .set using Fortran-initialized weights."
    )
    parser.add_argument(
        "--dataset-set",
        type=Path,
        default=DEFAULT_DATASETS_DIR / "cz84.set",
        help="Path to EEGLAB .set file.",
    )
    parser.add_argument(
        "--fortran-out",
        type=Path,
        default=None,
        help="Path to Fortran output directory. If omitted, auto-discovered from --fortran-search-root.",
    )
    parser.add_argument(
        "--fortran-search-root",
        type=Path,
        default=DEFAULT_BENCH_ROOT,
        help="Root to search for matching fortran_out when --fortran-out is omitted.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output run directory for Python artifacts.",
    )
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Leave unset to infer from Fortran W.",
    )
    parser.add_argument("--n-mixtures", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--optimizer", choices=OPTIMIZER_CHOICES, default="em")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--python-threads", type=int, default=1)
    args = parser.parse_args()

    resolved_fortran_out = resolve_fortran_out(
        dataset_set=args.dataset_set,
        fortran_out=args.fortran_out,
        fortran_search_root=args.fortran_search_root,
    )

    run_dir = args.run_dir if args.run_dir is not None else _default_run_dir(args.dataset_set, args.bench_root)
    manifest = run_python(
        dataset_set=args.dataset_set,
        fortran_out=resolved_fortran_out,
        run_dir=run_dir,
        max_iter=args.max_iter,
        n_components=args.n_components,
        n_mixtures=args.n_mixtures,
        random_state=args.random_state,
        device=args.device,
        verbose=args.verbose,
        python_threads=args.python_threads,
        optimizer=args.optimizer,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
