#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REAL_HOME = Path(os.path.expanduser("~")).resolve()
TMPDIR = Path(tempfile.gettempdir())
os.environ["HOME"] = str(TMPDIR)
os.environ["MPLCONFIGDIR"] = str(TMPDIR / "matplotlib")
os.environ["NUMBA_DISABLE_JIT"] = "1"

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
from matplotlib import pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
AMICA_SRC = REPO_ROOT / "amica-python" / "src"
if AMICA_SRC.exists() and str(AMICA_SRC) not in sys.path:
    sys.path.insert(0, str(AMICA_SRC))

from amica.utils.fortran import load_fortran_results

from mica_common import load_eeglab_set

DEFAULT_BENCH_ROOT = Path(__file__).resolve().parents[1] / "benchmark_runs"
DEFAULT_DATASETS_DIR = REAL_HOME / "amica_test_data" / "mica_release" / "datasets"


@dataclass
class CurveMetrics:
    n_iter: int
    ll_initial: float
    ll_final: float
    ll_peak: float
    ll_gain_to_final: float
    ll_gain_to_peak: float


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _clean_ll(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    arr = arr[arr != 0]
    return arr


def _curve_metrics(values: np.ndarray) -> CurveMetrics:
    arr = _clean_ll(values)
    if arr.size == 0:
        raise ValueError("LL curve is empty after removing zeros/non-finite values.")
    return CurveMetrics(
        n_iter=int(arr.size),
        ll_initial=float(arr[0]),
        ll_final=float(arr[-1]),
        ll_peak=float(np.max(arr)),
        ll_gain_to_final=float(arr[-1] - arr[0]),
        ll_gain_to_peak=float(np.max(arr) - arr[0]),
    )


def _corr_match_report(s_fortran: np.ndarray, s_python: np.ndarray) -> dict[str, float]:
    sf = (s_fortran - s_fortran.mean(axis=0, keepdims=True)) / (
        s_fortran.std(axis=0, keepdims=True) + 1e-12
    )
    sp = (s_python - s_python.mean(axis=0, keepdims=True)) / (
        s_python.std(axis=0, keepdims=True) + 1e-12
    )
    corr = np.abs((sf.T @ sp) / max(sf.shape[0] - 1, 1))
    row_ind, col_ind = linear_sum_assignment(-corr)
    matched = corr[row_ind, col_ind]
    return {
        "matched_source_abs_corr_mean": float(np.mean(matched)),
        "matched_source_abs_corr_median": float(np.median(matched)),
        "matched_source_abs_corr_min": float(np.min(matched)),
        "matched_source_abs_corr_max": float(np.max(matched)),
    }


def _compute_source_corr(
    *,
    dataset_set: Path,
    fortran_out: Path,
    python_npz: Path,
    n_components: int,
    n_mixtures: int,
) -> dict[str, float]:
    data = load_eeglab_set(dataset_set)
    fres = load_fortran_results(
        fortran_out,
        n_components=n_components,
        n_mixtures=n_mixtures,
        n_features=int(data.shape[1]),
    )
    pres = np.load(python_npz)
    fortran_components = fres["W"][:, :, 0] @ fres["S"][:n_components, :]
    python_components = np.asarray(pres["components"], dtype=np.float64)
    sf = data @ fortran_components.T
    sp = data @ python_components.T
    return _corr_match_report(sf, sp)


def _summarize_pair(
    *,
    dataset_name: str,
    fortran_dir: Path,
    python_dir: Path,
    datasets_dir: Path,
    compute_source_corr: bool,
) -> dict[str, object]:
    fortran_manifest = _read_json(fortran_dir / "fortran_run.json")
    python_manifest = _read_json(python_dir / "python_run.json")

    python_npz = np.load(python_dir / "python_results.npz")
    python_ll = _clean_ll(python_npz["ll"])
    fortran_results = load_fortran_results(
        fortran_dir / "fortran_out",
        n_components=int(python_manifest["n_components"]),
        n_mixtures=int(python_manifest["n_mixtures"]),
        n_features=int(python_manifest["data_shape_samples_features"][1]),
    )
    fortran_ll = _clean_ll(fortran_results["LL"])
    fortran_fit_seconds = float(fortran_manifest["fit_seconds"])
    python_fit_seconds = float(python_manifest["fit_seconds"])

    py_curve = _curve_metrics(python_ll)
    ft_curve = _curve_metrics(fortran_ll)

    shared_n = int(min(fortran_ll.size, python_ll.size))
    ll_delta = float(py_curve.ll_final - ft_curve.ll_final)

    record: dict[str, object] = {
        "dataset": dataset_name,
        "n_samples": int(python_manifest["data_shape_samples_features"][0]),
        "n_features": int(python_manifest["data_shape_samples_features"][1]),
        "n_components": int(python_manifest["n_components"]),
        "n_mixtures": int(python_manifest["n_mixtures"]),
        "fortran_final_ll": ft_curve.ll_final,
        "python_final_ll": py_curve.ll_final,
        "python_minus_fortran_ll": ll_delta,
        "abs_python_minus_fortran_ll": abs(ll_delta),
        "fortran_n_iter": ft_curve.n_iter,
        "python_n_iter": py_curve.n_iter,
        "python_iter_fraction_of_fortran": py_curve.n_iter / ft_curve.n_iter,
        "shared_n_iter": shared_n,
        "fortran_fit_seconds": fortran_fit_seconds,
        "python_fit_seconds": python_fit_seconds,
        "fortran_vs_python_speedup": (
            float(fortran_fit_seconds / python_fit_seconds)
            if python_fit_seconds > 0
            else None
        ),
    }

    if compute_source_corr:
        dataset_set = datasets_dir / f"{dataset_name}.set"
        record.update(
            _compute_source_corr(
                dataset_set=dataset_set,
                fortran_out=fortran_dir / "fortran_out",
                python_npz=python_dir / "python_results.npz",
                n_components=int(python_manifest["n_components"]),
                n_mixtures=int(python_manifest["n_mixtures"]),
            )
        )

    record["fortran_manifest"] = str(fortran_manifest["fortran_output_dir"])
    record["python_manifest"] = str(python_manifest["python_results_npz"])
    return record


def _env_summary(manifest: dict[str, object]) -> dict[str, object]:
    env = manifest.get("environment", {}) or {}
    if not isinstance(env, dict):
        env = {}
    slurm = env.get("slurm", {}) or {}
    if not isinstance(slurm, dict):
        slurm = {}
    return {
        "node": env.get("node_name") or env.get("hostname"),
        "cpuset": env.get("cpus_allowed_list"),
        "slurm_job_id": slurm.get("SLURM_JOB_ID"),
    }


def _read_fortran_ll(fortran_dir: Path, shape_manifest: dict[str, object]) -> np.ndarray:
    n_components = int(shape_manifest["n_components"])
    n_mixtures = int(shape_manifest["n_mixtures"])
    n_features = int(shape_manifest["data_shape_samples_features"][1])
    return _clean_ll(
        load_fortran_results(
            fortran_dir / "fortran_out",
            n_components=n_components,
            n_mixtures=n_mixtures,
            n_features=n_features,
        )["LL"]
    )


def _summarize_triplet_dataset(dataset_dir: Path) -> dict[str, object]:
    dataset_name = dataset_dir.name
    fortran_dir = dataset_dir / "fortran"
    em_dir = dataset_dir / "python_em"
    daarem_dir = dataset_dir / "python_daarem"

    fortran_manifest = _read_json(fortran_dir / "fortran_run.json")
    em_manifest = _read_json(em_dir / "python_run.json")
    daarem_manifest = _read_json(daarem_dir / "python_run.json")

    fortran_ll = _read_fortran_ll(fortran_dir, em_manifest)
    fortran_curve = _curve_metrics(fortran_ll)
    em_ll = _clean_ll(np.load(em_dir / "python_results.npz")["ll"])
    daarem_ll = _clean_ll(np.load(daarem_dir / "python_results.npz")["ll"])
    em_curve = _curve_metrics(em_ll)
    daarem_curve = _curve_metrics(daarem_ll)

    fortran_env = _env_summary(fortran_manifest)
    em_env = _env_summary(em_manifest)
    daarem_env = _env_summary(daarem_manifest)
    same_job = (
        fortran_env["slurm_job_id"]
        == em_env["slurm_job_id"]
        == daarem_env["slurm_job_id"]
    )
    same_node = fortran_env["node"] == em_env["node"] == daarem_env["node"]
    same_cpuset = fortran_env["cpuset"] == em_env["cpuset"] == daarem_env["cpuset"]

    fortran_fit_seconds = float(fortran_manifest["fit_seconds"])
    em_fit_seconds = float(em_manifest["fit_seconds"])
    daarem_fit_seconds = float(daarem_manifest["fit_seconds"])
    fortran_final_ll = float(em_manifest.get("fortran_ll_final", fortran_curve.ll_final))
    em_final_ll = float(em_manifest.get("python_ll_final", em_curve.ll_final))
    daarem_final_ll = float(daarem_manifest.get("python_ll_final", daarem_curve.ll_final))

    return {
        "dataset": dataset_name,
        "n_samples": int(em_manifest["data_shape_samples_features"][0]),
        "n_features": int(em_manifest["data_shape_samples_features"][1]),
        "n_components": int(em_manifest["n_components"]),
        "n_mixtures": int(em_manifest["n_mixtures"]),
        "fortran_final_ll": fortran_final_ll,
        "em_final_ll": em_final_ll,
        "daarem_final_ll": daarem_final_ll,
        "fortran_n_iter": int(fortran_manifest.get("fortran_n_iter") or fortran_curve.n_iter),
        "em_n_iter": int(em_manifest.get("python_n_iter") or em_curve.n_iter),
        "daarem_n_iter": int(daarem_manifest.get("python_n_iter") or daarem_curve.n_iter),
        "fortran_fit_seconds": fortran_fit_seconds,
        "em_fit_seconds": em_fit_seconds,
        "daarem_fit_seconds": daarem_fit_seconds,
        "em_fortran_seconds_ratio": em_fit_seconds / fortran_fit_seconds,
        "daarem_fortran_seconds_ratio": daarem_fit_seconds / fortran_fit_seconds,
        "daarem_em_seconds_ratio": daarem_fit_seconds / em_fit_seconds,
        "em_minus_fortran_ll": em_final_ll - fortran_final_ll,
        "daarem_minus_fortran_ll": daarem_final_ll - fortran_final_ll,
        "daarem_minus_em_ll": daarem_final_ll - em_final_ll,
        "same_slurm_job_id": bool(same_job),
        "same_node": bool(same_node),
        "same_cpuset": bool(same_cpuset),
        "node": fortran_env["node"],
        "cpuset": fortran_env["cpuset"],
        "slurm_job_id": fortran_env["slurm_job_id"],
        "fortran_manifest": str(fortran_dir / "fortran_run.json"),
        "em_manifest": str(em_dir / "python_run.json"),
        "daarem_manifest": str(daarem_dir / "python_run.json"),
    }


def _triplet_dataset_dirs(triplet_batch_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in triplet_batch_dir.iterdir()
        if p.is_dir()
        and (p / "fortran" / "fortran_run.json").exists()
        and (p / "python_em" / "python_run.json").exists()
        and (p / "python_daarem" / "python_run.json").exists()
    )


def _summarize_triplet_batch(triplet_batch_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_dir in _triplet_dataset_dirs(triplet_batch_dir):
        print(f"Summarizing {triplet_batch_dir.name}/{dataset_dir.name}...", file=sys.stderr)
        row = _summarize_triplet_dataset(dataset_dir)
        row["run"] = triplet_batch_dir.name
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No triplet dataset directories were found in {triplet_batch_dir}.")
    return pd.DataFrame(rows)


def _aggregate_triplet_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    mean_cols = [
        "fortran_final_ll",
        "em_final_ll",
        "daarem_final_ll",
        "fortran_n_iter",
        "em_n_iter",
        "daarem_n_iter",
        "fortran_fit_seconds",
        "em_fit_seconds",
        "daarem_fit_seconds",
        "em_fortran_seconds_ratio",
        "daarem_fortran_seconds_ratio",
        "daarem_em_seconds_ratio",
        "em_minus_fortran_ll",
        "daarem_minus_fortran_ll",
        "daarem_minus_em_ll",
    ]
    static_cols = ["n_samples", "n_features", "n_components", "n_mixtures"]
    bool_cols = ["same_slurm_job_id", "same_node", "same_cpuset"]

    rows: list[dict[str, object]] = []
    for dataset, group in run_df.groupby("dataset", sort=True):
        row: dict[str, object] = {"dataset": dataset, "n_runs": int(group["run"].nunique())}
        for col in static_cols:
            row[col] = int(group[col].iloc[0])
        for col in mean_cols:
            row[col] = float(group[col].mean())
        for col in ["fortran_fit_seconds", "em_fit_seconds", "daarem_fit_seconds"]:
            row[f"{col}_min"] = float(group[col].min())
            row[f"{col}_max"] = float(group[col].max())
        for col in bool_cols:
            row[col] = bool(group[col].all())
        row["node"] = ",".join(sorted(str(v) for v in group["node"].dropna().unique()))
        row["cpuset"] = ",".join(sorted(str(v) for v in group["cpuset"].dropna().unique()))
        row["slurm_job_id"] = ",".join(sorted(str(v) for v in group["slurm_job_id"].dropna().unique()))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def _plot_ll_parity(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    if "em_final_ll" in df.columns:
        ax.scatter(
            df["fortran_final_ll"],
            df["em_final_ll"],
            s=50,
            color="#5472E4",
            label="AMICA-Python",
        )
        ax.scatter(
            df["fortran_final_ll"],
            df["daarem_final_ll"],
            s=50,
            color="#D66A2C",
            marker="^",
            label="AMICA-Python (DAAREM)",
        )
        y_col = "em_final_ll"
        y_label = "AMICA-Python final LL"
    else:
        ax.scatter(df["fortran_final_ll"], df["python_final_ll"], s=50, color="#5472E4")
        y_col = "python_final_ll"
        y_label = "Python final LL"
    for row in df.itertuples(index=False):
        ax.annotate(row.dataset, (row.fortran_final_ll, getattr(row, y_col)), xytext=(4, 4), textcoords="offset points")
    ll_cols = ["fortran_final_ll", y_col]
    if "daarem_final_ll" in df.columns:
        ll_cols.append("daarem_final_ll")
    finite = np.concatenate([df[col].to_numpy() for col in ll_cols])
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="0.5")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Fortran final LL")
    ax.set_ylabel(y_label)
    ax.set_title("Final Log-Likelihood Parity")
    if "em_final_ll" in df.columns:
        ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def _plot_ll_delta(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if "em_minus_fortran_ll" in df.columns:
        plot_df = df.sort_values("daarem_minus_em_ll")
        x = np.arange(len(plot_df))
        width = 0.36
        ax.bar(
            x - width / 2,
            plot_df["em_minus_fortran_ll"],
            width=width,
            color="#5472E4",
            label="EM - Fortran",
        )
        ax.bar(
            x + width / 2,
            plot_df["daarem_minus_em_ll"],
            width=width,
            color="#D66A2C",
            label="DAAREM - EM",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["dataset"], rotation=45)
        ax.set_ylabel("Final LL difference")
        ax.legend(frameon=False)
    else:
        plot_df = df.sort_values("python_minus_fortran_ll")
        colors = np.where(
            plot_df["abs_python_minus_fortran_ll"] > 1e-2,
            "#d62728",
            "#2ca02c",
        )
        ax.bar(plot_df["dataset"], plot_df["python_minus_fortran_ll"], color=colors)
        ax.set_ylabel("Python - Fortran final LL")
        ax.tick_params(axis="x", rotation=45)
    ax.axhline(0.0, color="0.3", linewidth=1)
    ax.set_title("Per-Dataset Final LL Difference")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def _plot_runtime_comparison(df: pd.DataFrame, out_path: Path) -> None:
    if "em_fit_seconds" in df.columns:
        required_cols = ["fortran_fit_seconds", "em_fit_seconds", "daarem_fit_seconds"]
        sort_col = "daarem_fortran_seconds_ratio"
    else:
        required_cols = ["python_fit_seconds", "fortran_fit_seconds"]
        sort_col = "fortran_vs_python_speedup"
    plot_df = df.dropna(subset=required_cols).copy()
    plot_df = plot_df.sort_values(sort_col)
    x = np.arange(len(plot_df))
    if "em_fit_seconds" in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
        comparisons = [
            (axes[0], "em_fit_seconds", "AMICA-Python", "#5472E4"),
            (axes[1], "daarem_fit_seconds", "AMICA-Python (DAAREM)", "#D66A2C"),
        ]
        width = 0.38
        for ax, python_col, python_label, python_color in comparisons:
            fortran_yerr = _runtime_range_yerr(plot_df, "fortran_fit_seconds")
            python_yerr = _runtime_range_yerr(plot_df, python_col)
            ax.bar(
                x - width / 2,
                plot_df["fortran_fit_seconds"],
                yerr=fortran_yerr,
                capsize=3 if fortran_yerr is not None else 0,
                width=width,
                label="Fortran",
                color="#454843",
            )
            ax.bar(
                x + width / 2,
                plot_df[python_col],
                yerr=python_yerr,
                capsize=3 if python_yerr is not None else 0,
                width=width,
                label=python_label,
                color=python_color,
            )
            ax.set_ylabel("Wall time (seconds)")
            ax.set_title(f"Fortran vs {python_label}")
            ax.legend(frameon=False)
            ax.grid(True, axis="y", alpha=0.25)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(plot_df["dataset"], rotation=45, ha="right")
        fig.suptitle("Runtime Comparison")
    else:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        width = 0.42
        ax.bar(x - width / 2, plot_df["fortran_fit_seconds"], width=width, label="Fortran (sec)", color="#454843")
        ax.bar(x + width / 2, plot_df["python_fit_seconds"], width=width, label="Python (sec)", color="#7D66D9")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["dataset"], rotation=45)
        ax.set_ylabel("Wall time (seconds)")
        ax.set_title("Runtime Comparison")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def _runtime_range_yerr(df: pd.DataFrame, col: str) -> np.ndarray | None:
    min_col = f"{col}_min"
    max_col = f"{col}_max"
    if min_col not in df.columns or max_col not in df.columns:
        return None
    lower = df[col].to_numpy(dtype=float) - df[min_col].to_numpy(dtype=float)
    upper = df[max_col].to_numpy(dtype=float) - df[col].to_numpy(dtype=float)
    return np.vstack([lower, upper])


def _plot_convergence_examples(
    *,
    df: pd.DataFrame,
    fortran_batch_dir: Path,
    python_batch_dir: Path,
    out_path: Path,
) -> None:
    subjects = list(df.sort_values("abs_python_minus_fortran_ll", ascending=False)["dataset"].head(4))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=False)
    for ax, dataset in zip(axes.flat, subjects):
        py_ll = _clean_ll(np.load(python_batch_dir / dataset / "python_results.npz")["ll"])
        ft_ll = _clean_ll(
            load_fortran_results(
                fortran_batch_dir / dataset / "fortran_out",
                n_components=int(df.loc[df["dataset"] == dataset, "n_components"].iloc[0]),
                n_mixtures=int(df.loc[df["dataset"] == dataset, "n_mixtures"].iloc[0]),
                n_features=int(df.loc[df["dataset"] == dataset, "n_features"].iloc[0]),
            )["LL"]
        )
        ax.plot(np.arange(1, ft_ll.size + 1), ft_ll, label="Fortran", color="#1f77b4")
        ax.plot(np.arange(1, py_ll.size + 1), py_ll, label="Python", color="#ff7f0e")
        ax.set_title(dataset)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("LL")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Convergence Curves: Largest Final-LL Differences", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_triplet_convergence_examples(
    *,
    df: pd.DataFrame,
    triplet_batch_dir: Path,
    out_path: Path,
) -> None:
    subjects = list(df.sort_values("daarem_minus_em_ll")["dataset"].head(4))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=False)
    for ax, dataset in zip(axes.flat, subjects):
        dataset_dir = triplet_batch_dir / dataset
        ft_ll = _clean_ll(
            load_fortran_results(
                dataset_dir / "fortran" / "fortran_out",
                n_components=int(df.loc[df["dataset"] == dataset, "n_components"].iloc[0]),
                n_mixtures=int(df.loc[df["dataset"] == dataset, "n_mixtures"].iloc[0]),
                n_features=int(df.loc[df["dataset"] == dataset, "n_features"].iloc[0]),
            )["LL"]
        )
        em_ll = _clean_ll(np.load(dataset_dir / "python_em" / "python_results.npz")["ll"])
        daarem_ll = _clean_ll(np.load(dataset_dir / "python_daarem" / "python_results.npz")["ll"])
        ax.plot(np.arange(1, ft_ll.size + 1), ft_ll, label="Fortran", color="#454843")
        ax.plot(np.arange(1, em_ll.size + 1), em_ll, label="EM", color="#5472E4")
        ax.plot(np.arange(1, daarem_ll.size + 1), daarem_ll, label="DAAREM", color="#D66A2C")
        ax.set_title(dataset)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("LL")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Convergence Curves: Largest DAAREM Final-LL Drifts", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_markdown(df: pd.DataFrame, aggregate: dict[str, object]) -> str:
    if "em_final_ll" in df.columns:
        display_cols = [
            "dataset",
            "fortran_n_iter",
            "em_n_iter",
            "daarem_n_iter",
            "fortran_fit_seconds",
            "em_fit_seconds",
            "daarem_fit_seconds",
            "em_fortran_seconds_ratio",
            "daarem_fortran_seconds_ratio",
            "daarem_em_seconds_ratio",
            "em_minus_fortran_ll",
            "daarem_minus_em_ll",
            "same_slurm_job_id",
            "same_node",
            "same_cpuset",
        ]
        table = (
            df.sort_values("daarem_minus_em_ll")[display_cols]
            .round(6)
            .to_markdown(index=False)
        )
        lines = [
            "# Triplet Benchmark Summary",
            "",
            f"- Datasets compared: {aggregate['n_datasets']}",
            f"- Same SLURM job: {aggregate['same_slurm_job_id_count']}/{aggregate['n_datasets']}",
            f"- Same node: {aggregate['same_node_count']}/{aggregate['n_datasets']}",
            f"- Same cpuset: {aggregate['same_cpuset_count']}/{aggregate['n_datasets']}",
            f"- Median EM/Fortran wall-time ratio: {aggregate['median_em_fortran_seconds_ratio']:.3f}",
            f"- Median DAAREM/Fortran wall-time ratio: {aggregate['median_daarem_fortran_seconds_ratio']:.3f}",
            f"- Median DAAREM/EM wall-time ratio: {aggregate['median_daarem_em_seconds_ratio']:.3f}",
            f"- Median EM - Fortran final LL: {aggregate['median_em_minus_fortran_ll']:.6g}",
            f"- Median DAAREM - EM final LL: {aggregate['median_daarem_minus_em_ll']:.6g}",
            f"- DAAREM below EM by >1e-4: {aggregate['daarem_below_em_gt_1e_4_count']}/{aggregate['n_datasets']}",
            f"- DAAREM below EM by >1e-3: {aggregate['daarem_below_em_gt_1e_3_count']}/{aggregate['n_datasets']}",
        ]
        lines.extend(["", table, ""])
        return "\n".join(lines)

    display_cols = [
        "dataset",
        "fortran_final_ll",
        "python_final_ll",
        "python_minus_fortran_ll",
        "fortran_n_iter",
        "python_n_iter",
    ]
    if "matched_source_abs_corr_mean" in df.columns:
        display_cols.append("matched_source_abs_corr_mean")
    if "fortran_fit_seconds" in df.columns and "python_fit_seconds" in df.columns:
        display_cols.extend(["fortran_fit_seconds", "python_fit_seconds", "fortran_vs_python_speedup"])

    table = (
        df.sort_values("abs_python_minus_fortran_ll", ascending=False)[display_cols]
        .round(6)
        .to_markdown(index=False)
    )
    lines = [
        "# Benchmark Summary",
        "",
        f"- Datasets compared: {aggregate['n_datasets']}",
        f"- Median |Python - Fortran| final LL: {aggregate['median_abs_final_ll_delta']:.6g}",
        f"- Mean |Python - Fortran| final LL excluding worst case: {aggregate['mean_abs_final_ll_delta_excluding_worst']:.6g}",
        f"- Worst-case dataset by final LL delta: {aggregate['worst_case_dataset']}",
    ]
    if "mean_matched_source_abs_corr" in aggregate:
        lines.append(
            f"- Mean matched-source abs corr: {aggregate['mean_matched_source_abs_corr']:.6f}"
        )
    if "median_fortran_vs_python_speedup" in aggregate:
        lines.append(
            f"- Median Fortran/Python speed ratio: {aggregate['median_fortran_vs_python_speedup']:.3f}x"
        )
    lines.extend(["", table, ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize AMICA Python vs Fortran benchmark batches."
    )
    parser.add_argument("--triplet-batch-dir", type=Path, default=None)
    parser.add_argument(
        "--triplet-batch-glob",
        default=None,
        help="Glob for multiple triplet batch directories to average before plotting.",
    )
    parser.add_argument("--fortran-batch-dir", type=Path, default=None)
    parser.add_argument("--python-batch-dir", type=Path, default=None)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-source-corr", action="store_true")
    args = parser.parse_args()

    if args.triplet_batch_glob is not None:
        if (
            args.triplet_batch_dir is not None
            or args.fortran_batch_dir is not None
            or args.python_batch_dir is not None
        ):
            parser.error(
                "--triplet-batch-glob cannot be combined with --triplet-batch-dir "
                "or separate batch dirs."
            )
        triplet_batch_dirs = [
            Path(p).expanduser().resolve()
            for p in sorted(glob.glob(os.path.expanduser(args.triplet_batch_glob)))
            if Path(p).is_dir()
        ]
        if not triplet_batch_dirs:
            raise FileNotFoundError(f"No directories matched {args.triplet_batch_glob!r}.")
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else Path("results/benchmark_summary").resolve()
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        run_df = pd.concat(
            [_summarize_triplet_batch(batch_dir) for batch_dir in triplet_batch_dirs],
            ignore_index=True,
        )
        df = _aggregate_triplet_runs(run_df)
        unpaired_df = run_df[
            ~(run_df["same_slurm_job_id"] & run_df["same_node"] & run_df["same_cpuset"])
        ]
        if not unpaired_df.empty:
            failed = ", ".join(
                f"{row.run}/{row.dataset}" for row in unpaired_df.itertuples(index=False)
            )
            print(
                f"WARNING: triplet pairing check failed for {len(unpaired_df)} run dataset(s): {failed}",
                file=sys.stderr,
            )

        aggregate: dict[str, object] = {
            "triplet_batch_dirs": [str(path) for path in triplet_batch_dirs],
            "n_runs": int(len(triplet_batch_dirs)),
            "n_datasets": int(len(df)),
            "n_run_datasets": int(len(run_df)),
            "same_slurm_job_id_count": int(df["same_slurm_job_id"].sum()),
            "same_node_count": int(df["same_node"].sum()),
            "same_cpuset_count": int(df["same_cpuset"].sum()),
            "median_em_fortran_seconds_ratio": float(df["em_fortran_seconds_ratio"].median()),
            "mean_em_fortran_seconds_ratio": float(df["em_fortran_seconds_ratio"].mean()),
            "median_daarem_fortran_seconds_ratio": float(df["daarem_fortran_seconds_ratio"].median()),
            "mean_daarem_fortran_seconds_ratio": float(df["daarem_fortran_seconds_ratio"].mean()),
            "median_daarem_em_seconds_ratio": float(df["daarem_em_seconds_ratio"].median()),
            "mean_daarem_em_seconds_ratio": float(df["daarem_em_seconds_ratio"].mean()),
            "em_faster_than_fortran_count": int((df["em_fortran_seconds_ratio"] < 1.0).sum()),
            "daarem_faster_than_fortran_count": int((df["daarem_fortran_seconds_ratio"] < 1.0).sum()),
            "daarem_faster_than_em_count": int((df["daarem_em_seconds_ratio"] < 1.0).sum()),
            "daarem_fewer_iter_than_em_count": int((df["daarem_n_iter"] < df["em_n_iter"]).sum()),
            "median_em_minus_fortran_ll": float(df["em_minus_fortran_ll"].median()),
            "median_daarem_minus_fortran_ll": float(df["daarem_minus_fortran_ll"].median()),
            "median_daarem_minus_em_ll": float(df["daarem_minus_em_ll"].median()),
            "em_below_fortran_gt_1e_4_count": int((df["em_minus_fortran_ll"] < -1e-4).sum()),
            "daarem_below_em_gt_1e_4_count": int((df["daarem_minus_em_ll"] < -1e-4).sum()),
            "daarem_below_em_gt_1e_3_count": int((df["daarem_minus_em_ll"] < -1e-3).sum()),
        }

        csv_path = output_dir / "benchmark_summary.csv"
        run_csv_path = output_dir / "benchmark_summary_runs.csv"
        json_path = output_dir / "benchmark_summary.json"
        md_path = output_dir / "benchmark_summary.md"
        parity_png = output_dir / "final_ll_parity.png"
        delta_png = output_dir / "final_ll_delta.png"
        timing_png = output_dir / "runtime_comparison.png"

        df.to_csv(csv_path, index=False)
        run_df.sort_values(["run", "dataset"]).to_csv(run_csv_path, index=False)
        json_path.write_text(
            json.dumps(
                {
                    "aggregate": aggregate,
                    "datasets": df.to_dict(orient="records"),
                    "run_datasets": run_df.sort_values(["run", "dataset"]).to_dict(orient="records"),
                },
                indent=2,
            )
            + "\n"
        )
        md_path.write_text(_make_markdown(df, aggregate) + "\n")
        _plot_ll_parity(df, parity_png)
        _plot_ll_delta(df, delta_png)
        _plot_runtime_comparison(df, timing_png)

        print(json.dumps({"aggregate": aggregate, "output_dir": str(output_dir)}, indent=2))
        return

    if args.triplet_batch_dir is not None:
        if args.fortran_batch_dir is not None or args.python_batch_dir is not None:
            parser.error("--triplet-batch-dir cannot be combined with separate batch dirs.")
        triplet_batch_dir = args.triplet_batch_dir.expanduser().resolve()
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else triplet_batch_dir / "summary_outputs"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dirs = _triplet_dataset_dirs(triplet_batch_dir)
        if not dataset_dirs:
            raise FileNotFoundError("No triplet dataset directories were found.")

        rows: list[dict[str, object]] = []
        for dataset_dir in dataset_dirs:
            print(f"Summarizing {dataset_dir.name}...", file=sys.stderr)
            rows.append(_summarize_triplet_dataset(dataset_dir))

        df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
        unpaired_df = df[
            ~(df["same_slurm_job_id"] & df["same_node"] & df["same_cpuset"])
        ]
        if not unpaired_df.empty:
            failed = ", ".join(unpaired_df["dataset"].astype(str))
            print(
                f"WARNING: triplet pairing check failed for {len(unpaired_df)} dataset(s): {failed}",
                file=sys.stderr,
            )

        aggregate: dict[str, object] = {
            "triplet_batch_dir": str(triplet_batch_dir),
            "n_datasets": int(len(df)),
            "same_slurm_job_id_count": int(df["same_slurm_job_id"].sum()),
            "same_node_count": int(df["same_node"].sum()),
            "same_cpuset_count": int(df["same_cpuset"].sum()),
            "median_em_fortran_seconds_ratio": float(df["em_fortran_seconds_ratio"].median()),
            "mean_em_fortran_seconds_ratio": float(df["em_fortran_seconds_ratio"].mean()),
            "median_daarem_fortran_seconds_ratio": float(df["daarem_fortran_seconds_ratio"].median()),
            "mean_daarem_fortran_seconds_ratio": float(df["daarem_fortran_seconds_ratio"].mean()),
            "median_daarem_em_seconds_ratio": float(df["daarem_em_seconds_ratio"].median()),
            "mean_daarem_em_seconds_ratio": float(df["daarem_em_seconds_ratio"].mean()),
            "em_faster_than_fortran_count": int((df["em_fortran_seconds_ratio"] < 1.0).sum()),
            "daarem_faster_than_fortran_count": int((df["daarem_fortran_seconds_ratio"] < 1.0).sum()),
            "daarem_faster_than_em_count": int((df["daarem_em_seconds_ratio"] < 1.0).sum()),
            "daarem_fewer_iter_than_em_count": int((df["daarem_n_iter"] < df["em_n_iter"]).sum()),
            "median_em_minus_fortran_ll": float(df["em_minus_fortran_ll"].median()),
            "median_daarem_minus_fortran_ll": float(df["daarem_minus_fortran_ll"].median()),
            "median_daarem_minus_em_ll": float(df["daarem_minus_em_ll"].median()),
            "em_below_fortran_gt_1e_4_count": int((df["em_minus_fortran_ll"] < -1e-4).sum()),
            "daarem_below_em_gt_1e_4_count": int((df["daarem_minus_em_ll"] < -1e-4).sum()),
            "daarem_below_em_gt_1e_3_count": int((df["daarem_minus_em_ll"] < -1e-3).sum()),
        }

        csv_path = output_dir / "benchmark_summary.csv"
        json_path = output_dir / "benchmark_summary.json"
        md_path = output_dir / "benchmark_summary.md"
        parity_png = output_dir / "final_ll_parity.png"
        delta_png = output_dir / "final_ll_delta.png"
        conv_png = output_dir / "convergence_examples.png"
        timing_png = output_dir / "runtime_comparison.png"

        df.to_csv(csv_path, index=False)
        json_path.write_text(
            json.dumps(
                {
                    "aggregate": aggregate,
                    "datasets": df.to_dict(orient="records"),
                },
                indent=2,
            )
            + "\n"
        )
        md_path.write_text(_make_markdown(df, aggregate) + "\n")
        _plot_ll_parity(df, parity_png)
        _plot_ll_delta(df, delta_png)
        _plot_triplet_convergence_examples(
            df=df,
            triplet_batch_dir=triplet_batch_dir,
            out_path=conv_png,
        )
        _plot_runtime_comparison(df, timing_png)

        print(json.dumps({"aggregate": aggregate, "output_dir": str(output_dir)}, indent=2))
        return

    if args.fortran_batch_dir is None or args.python_batch_dir is None:
        parser.error("Provide either --triplet-batch-dir or both --fortran-batch-dir and --python-batch-dir.")

    fortran_batch_dir = args.fortran_batch_dir.expanduser().resolve()
    python_batch_dir = args.python_batch_dir.expanduser().resolve()
    datasets_dir = args.datasets_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else python_batch_dir / "summary_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = sorted(
        {
            p.parent.name
            for p in python_batch_dir.glob("*/python_run.json")
            if (fortran_batch_dir / p.parent.name / "fortran_run.json").exists()
        }
    )
    if not dataset_names:
        raise FileNotFoundError("No paired Python/Fortran dataset directories were found.")

    rows: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        print(f"Summarizing {dataset_name}...", file=sys.stderr)
        rows.append(
            _summarize_pair(
                dataset_name=dataset_name,
                fortran_dir=fortran_batch_dir / dataset_name,
                python_dir=python_batch_dir / dataset_name,
                datasets_dir=datasets_dir,
                compute_source_corr=not args.skip_source_corr,
            )
        )

    df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    worst_row = df.iloc[df["abs_python_minus_fortran_ll"].argmax()]

    aggregate: dict[str, object] = {
        "fortran_batch_dir": str(fortran_batch_dir),
        "python_batch_dir": str(python_batch_dir),
        "datasets_dir": str(datasets_dir),
        "n_datasets": int(len(df)),
        "median_abs_final_ll_delta": float(df["abs_python_minus_fortran_ll"].median()),
        "mean_abs_final_ll_delta": float(df["abs_python_minus_fortran_ll"].mean()),
        "mean_abs_final_ll_delta_excluding_worst": float(
            df["abs_python_minus_fortran_ll"].sort_values().iloc[:-1].mean()
        ),
        "max_abs_final_ll_delta": float(df["abs_python_minus_fortran_ll"].max()),
        "worst_case_dataset": str(worst_row["dataset"]),
        "median_python_iter_fraction_of_fortran": float(df["python_iter_fraction_of_fortran"].median()),
    }
    if "matched_source_abs_corr_mean" in df.columns:
        aggregate["mean_matched_source_abs_corr"] = float(df["matched_source_abs_corr_mean"].mean())
    timing_df = df.dropna(subset=["python_fit_seconds", "fortran_fit_seconds"])
    if not timing_df.empty:
        aggregate["median_python_fit_seconds"] = float(timing_df["python_fit_seconds"].median())
        aggregate["median_fortran_fit_seconds"] = float(timing_df["fortran_fit_seconds"].median())
        aggregate["median_fortran_vs_python_speedup"] = float(timing_df["fortran_vs_python_speedup"].median())
        aggregate["mean_fortran_vs_python_speedup"] = float(timing_df["fortran_vs_python_speedup"].mean())

    csv_path = output_dir / "benchmark_summary.csv"
    json_path = output_dir / "benchmark_summary.json"
    md_path = output_dir / "benchmark_summary.md"
    parity_png = output_dir / "final_ll_parity.png"
    delta_png = output_dir / "final_ll_delta.png"
    conv_png = output_dir / "convergence_examples.png"
    timing_png = output_dir / "runtime_comparison.png"

    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "aggregate": aggregate,
                "datasets": df.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n"
    )
    md_path.write_text(_make_markdown(df, aggregate) + "\n")

    _plot_ll_parity(df, parity_png)
    _plot_ll_delta(df, delta_png)
    _plot_convergence_examples(
        df=df,
        fortran_batch_dir=fortran_batch_dir,
        python_batch_dir=python_batch_dir,
        out_path=conv_png,
    )
    _plot_runtime_comparison(df, timing_png)

    print(json.dumps({"aggregate": aggregate, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
