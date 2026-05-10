#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from amica import AMICA
from amica.utils.fortran import load_fortran_results, load_initial_weights

from mica_common import (
    clean_ll,
    collect_environment_metadata,
    discover_fortran_out,
    infer_fortran_n_components,
    load_eeglab_set,
)


DEFAULT_BENCH_ROOT = Path(__file__).resolve().parents[1] / "benchmark_runs"
DEFAULT_DATASETS_DIR = Path.home() / "amica_test_data" / "mica_release" / "datasets"

BASE_FIELDS = [
    "dataset",
    "label",
    "optimizer",
    "config_index",
    "run_index",
    "random_state",
    "fit_seconds",
    "max_iter",
    "n_iter",
    "converged",
    "final_log_likelihood",
    "peak_log_likelihood",
    "initial_log_likelihood",
    "ll_gain_to_final",
    "ll_gain_to_peak",
    "last_delta_ll",
    "mean_delta_ll_last_5",
    "n_ll_decreases",
    "min_delta_ll",
    "fortran_final_log_likelihood",
    "python_minus_fortran_final_ll",
    "accelerator_order",
    "accelerator_start_iter",
    "accelerator_period",
    "accelerator_eps_monotone",
    "accelerator_validate_candidate",
    "accelerator_max_restarts",
    "accelerator_damping",
    "accelerator_ridge",
    "accelerator_daarem_alpha",
    "accelerator_daarem_kappa",
    "accelerator_cycl_monotone_tol",
]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_bool_list(value: str) -> list[bool]:
    parsed: list[bool] = []
    for item in value.split(","):
        lowered = item.strip().lower()
        if not lowered:
            continue
        if lowered in {"1", "true", "yes", "y"}:
            parsed.append(True)
        elif lowered in {"0", "false", "no", "n"}:
            parsed.append(False)
        else:
            raise ValueError(f"Could not parse boolean value: {item!r}")
    return parsed


def ll_at_iteration(ll: np.ndarray, iteration: int) -> float:
    if iteration < 1 or ll.size < iteration:
        return float("nan")
    return float(ll[iteration - 1])


def format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.10f}"


def row_for_csv(row: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    formatted = row.copy()
    for key in fieldnames:
        value = formatted.get(key)
        if isinstance(value, float):
            formatted[key] = format_float(value)
    return formatted


def summarize_ll(
    ll_values: np.ndarray,
    *,
    max_iter: int,
    probes: list[int],
    fortran_final_ll: float,
) -> dict[str, Any]:
    ll = clean_ll(ll_values)
    if ll.size == 0:
        row: dict[str, Any] = {
            "max_iter": int(max_iter),
            "n_iter": 0,
            "converged": "no",
            "final_log_likelihood": float("nan"),
            "peak_log_likelihood": float("nan"),
            "initial_log_likelihood": float("nan"),
            "ll_gain_to_final": float("nan"),
            "ll_gain_to_peak": float("nan"),
            "last_delta_ll": float("nan"),
            "mean_delta_ll_last_5": float("nan"),
            "n_ll_decreases": 0,
            "min_delta_ll": float("nan"),
            "fortran_final_log_likelihood": fortran_final_ll,
            "python_minus_fortran_final_ll": float("nan"),
        }
    else:
        deltas = np.diff(ll)
        last_deltas = deltas[-5:]
        final_ll = float(ll[-1])
        n_iter = int(ll.size)
        row = {
            "max_iter": int(max_iter),
            "n_iter": n_iter,
            "converged": "yes" if n_iter < int(max_iter) else "no",
            "final_log_likelihood": final_ll,
            "peak_log_likelihood": float(np.max(ll)),
            "initial_log_likelihood": float(ll[0]),
            "ll_gain_to_final": float(ll[-1] - ll[0]),
            "ll_gain_to_peak": float(np.max(ll) - ll[0]),
            "last_delta_ll": float(deltas[-1]) if deltas.size else float("nan"),
            "mean_delta_ll_last_5": (
                float(np.mean(last_deltas)) if last_deltas.size else float("nan")
            ),
            "n_ll_decreases": int(np.count_nonzero(deltas < 0.0)),
            "min_delta_ll": float(np.min(deltas)) if deltas.size else float("nan"),
            "fortran_final_log_likelihood": fortran_final_ll,
            "python_minus_fortran_final_ll": final_ll - fortran_final_ll,
        }
    for iteration in probes:
        row[f"ll_at_{iteration}"] = ll_at_iteration(ll, iteration)
    return row


def write_rows(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(row_for_csv(row, fieldnames) for row in rows)


def build_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for (
        order,
        start_iter,
        period,
        eps_monotone,
        validate_candidate,
        max_restarts,
        damping,
        ridge,
        daarem_alpha,
        daarem_kappa,
        cycl_monotone_tol,
    ) in itertools.product(
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
    ):
        configs.append(
            {
                "accelerator_order": order,
                "accelerator_start_iter": start_iter,
                "accelerator_period": period,
                "accelerator_eps_monotone": eps_monotone,
                "accelerator_validate_candidate": validate_candidate,
                "accelerator_max_restarts": max_restarts,
                "accelerator_damping": damping,
                "accelerator_ridge": ridge,
                "accelerator_daarem_alpha": daarem_alpha,
                "accelerator_daarem_kappa": daarem_kappa,
                "accelerator_cycl_monotone_tol": cycl_monotone_tol,
            }
        )
    return configs


def configure_threads(python_threads: int) -> None:
    thread_count = str(int(python_threads))
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count
    torch.set_num_threads(int(python_threads))


def fit_once(
    *,
    data: np.ndarray,
    n_components: int,
    n_mixtures: int,
    max_iter: int,
    random_state: int,
    device: str,
    verbose: int,
    optimizer_kwargs: dict[str, object],
    w_init: np.ndarray,
    sbeta_init: np.ndarray,
    mu_init: np.ndarray,
) -> tuple[float, np.ndarray]:

    model = AMICA(
        n_components=int(n_components),
        n_mixtures=int(n_mixtures),
        max_iter=int(max_iter),
        random_state=int(random_state),
        device=device,
        verbose=int(verbose),
        optimizer="daarem",
        optimizer_kwargs=optimizer_kwargs,
        w_init=w_init,
        sbeta_init=sbeta_init,
        mu_init=mu_init,
    )
    t0 = time.perf_counter()
    model.fit(data)
    fit_seconds = time.perf_counter() - t0
    ll = np.asarray(model.ll_ if hasattr(model, "ll_") else model._ll, dtype=float)
    return fit_seconds, clean_ll(ll)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep AMICA-Python DAAREM hyperparameters for one mica_release dataset."
    )
    parser.add_argument("--dataset-set", type=Path, required=True)
    parser.add_argument("--fortran-out", type=Path, default=None)
    parser.add_argument("--fortran-search-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--n-mixtures", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--python-threads", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--label", default="")
    parser.add_argument("--accelerator-orders", type=parse_int_list, default=[1])
    parser.add_argument("--accelerator-start-iters", type=parse_int_list, default=[5])
    parser.add_argument("--accelerator-periods", type=parse_int_list, default=[1])
    parser.add_argument("--accelerator-eps-monotones", type=parse_float_list, default=[0.0])
    parser.add_argument(
        "--accelerator-validate-candidates",
        type=parse_bool_list,
        default=[True],
    )
    parser.add_argument("--accelerator-max-restarts", type=parse_int_list, default=[20])
    parser.add_argument("--accelerator-dampings", type=parse_float_list, default=[1.0])
    parser.add_argument("--accelerator-ridges", type=parse_float_list, default=[1e-8])
    parser.add_argument("--accelerator-daarem-alphas", type=parse_float_list, default=[1.2])
    parser.add_argument("--accelerator-daarem-kappas", type=parse_int_list, default=[25])
    parser.add_argument(
        "--accelerator-cycl-monotone-tols",
        type=parse_float_list,
        default=[0.0],
    )
    parser.add_argument("--ll-probes", type=parse_int_list, default=[25, 50, 100, 200])
    args = parser.parse_args()

    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1")

    dataset_set = args.dataset_set.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_threads(int(args.python_threads))
    probes = sorted(set(args.ll_probes))
    fieldnames = BASE_FIELDS + [f"ll_at_{iteration}" for iteration in probes]

    fortran_out = (
        args.fortran_out.expanduser().resolve()
        if args.fortran_out is not None
        else discover_fortran_out(
            dataset_set=dataset_set,
            search_root=args.fortran_search_root.expanduser().resolve(),
        )
    )
    data = load_eeglab_set(dataset_set)
    n_features = int(data.shape[1])
    n_components = (
        infer_fortran_n_components(fortran_out)
        if args.n_components is None
        else int(args.n_components)
    )
    n_mixtures = int(args.n_mixtures)
    w_init, sbeta_init, mu_init = load_initial_weights(
        fortran_out,
        n_components=n_components,
        n_mixtures=n_mixtures,
    )
    fortran_results = load_fortran_results(
        fortran_out,
        n_components=n_components,
        n_mixtures=n_mixtures,
        n_features=n_features,
    )
    fortran_ll = clean_ll(fortran_results["LL"])
    fortran_final_ll = float(fortran_ll[-1]) if fortran_ll.size else float("nan")

    configs = build_configs(args)
    rows: list[dict[str, Any]] = []
    ll_records: list[np.ndarray] = []
    ll_config_index: list[int] = []
    ll_run_index: list[int] = []

    for run_index in range(args.n_runs):
        random_state = int(args.random_state) + run_index
        for config_index, config in enumerate(configs):
            fit_seconds, ll = fit_once(
                data=data,
                n_components=n_components,
                n_mixtures=n_mixtures,
                max_iter=int(args.max_iter),
                random_state=random_state,
                device=args.device,
                verbose=int(args.verbose),
                optimizer_kwargs=config,
                w_init=w_init,
                sbeta_init=sbeta_init,
                mu_init=mu_init,
            )
            row = summarize_ll(
                ll,
                max_iter=int(args.max_iter),
                probes=probes,
                fortran_final_ll=fortran_final_ll,
            )
            row.update(
                {
                    "dataset": dataset_set.stem,
                    "label": args.label,
                    "optimizer": "daarem",
                    "config_index": config_index,
                    "run_index": run_index,
                    "random_state": random_state,
                    "fit_seconds": fit_seconds,
                    **config,
                }
            )
            rows.append(row)
            ll_records.append(ll)
            ll_config_index.append(config_index)
            ll_run_index.append(run_index)
            print(
                f"dataset={dataset_set.stem} config={config_index} run={run_index} "
                f"fit_seconds={fit_seconds:.6f} n_iter={row['n_iter']} "
                f"final_ll={row['final_log_likelihood']:.10f} kwargs={config}",
                flush=True,
            )

    csv_path = run_dir / "daarem_sweep.csv"
    write_rows(csv_path, rows, fieldnames)

    max_len = max((ll.size for ll in ll_records), default=0)
    ll_matrix = np.full((len(ll_records), max_len), np.nan, dtype=np.float64)
    for i, ll in enumerate(ll_records):
        ll_matrix[i, : ll.size] = ll
    ll_npz = run_dir / "daarem_ll_curves.npz"
    np.savez(
        ll_npz,
        ll=ll_matrix,
        config_index=np.asarray(ll_config_index, dtype=np.int64),
        run_index=np.asarray(ll_run_index, dtype=np.int64),
    )

    manifest = {
        "dataset_set": str(dataset_set),
        "fortran_output_dir": str(fortran_out),
        "run_dir": str(run_dir),
        "bench_root": str(args.bench_root.expanduser().resolve()),
        "data_shape_samples_features": [int(data.shape[0]), int(data.shape[1])],
        "max_iter": int(args.max_iter),
        "n_components": int(n_components),
        "n_mixtures": int(n_mixtures),
        "random_state": int(args.random_state),
        "n_runs": int(args.n_runs),
        "device": args.device,
        "optimizer": "daarem",
        "python_threads": int(args.python_threads),
        "environment": collect_environment_metadata(),
        "n_configs": len(configs),
        "n_fits": len(rows),
        "configs": configs,
        "results": rows,
        "csv": str(csv_path),
        "ll_curves_npz": str(ll_npz),
    }
    manifest_path = run_dir / "daarem_sweep_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
