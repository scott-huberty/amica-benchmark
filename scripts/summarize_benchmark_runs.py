#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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

ITER_RE = re.compile(
    r"iter\s+(?P<iter>\d+)\s+"
    r"lrate =\s+(?P<lrate>[-+0-9.Ee]+)\s+"
    r"LL =\s+(?P<ll>[-+0-9.Ee]+)\s+"
    r"nd =\s+(?P<nd>[-+0-9.Ee]+).*?"
    r"\(\s*(?P<seconds>[-+0-9.Ee]+)\s*s,"
)
FORTRAN_TOTAL_HOURS_RE = re.compile(r"Execution time:\s*(?P<hours>[-+0-9.Ee]+)\s*h")
PYTHON_TOTAL_SECONDS_RE = re.compile(r"Finished in\s+(?P<seconds>[-+0-9.Ee]+)\s+seconds")


@dataclass
class CurveMetrics:
    n_iter: int
    ll_initial: float
    ll_final: float
    ll_peak: float
    ll_gain_to_final: float
    ll_gain_to_peak: float
    iter_peak: int
    final_minus_peak: float
    iter_to_90pct_peak_gain: int | None
    iter_to_95pct_peak_gain: int | None
    diverged_after_peak: bool


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _clean_ll(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    arr = arr[arr != 0]
    return arr


def _iter_to_fraction(arr: np.ndarray, fraction: float) -> int | None:
    if arr.size == 0:
        return None
    start = float(arr[0])
    peak = float(np.max(arr))
    gain = peak - start
    if gain <= 0:
        return 1
    target = start + fraction * gain
    idx = np.flatnonzero(arr >= target)
    if idx.size == 0:
        return None
    return int(idx[0] + 1)


def _curve_metrics(values: np.ndarray) -> CurveMetrics:
    arr = _clean_ll(values)
    if arr.size == 0:
        raise ValueError("LL curve is empty after removing zeros/non-finite values.")
    peak_idx = int(np.argmax(arr))
    return CurveMetrics(
        n_iter=int(arr.size),
        ll_initial=float(arr[0]),
        ll_final=float(arr[-1]),
        ll_peak=float(arr[peak_idx]),
        ll_gain_to_final=float(arr[-1] - arr[0]),
        ll_gain_to_peak=float(arr[peak_idx] - arr[0]),
        iter_peak=int(peak_idx + 1),
        final_minus_peak=float(arr[-1] - arr[peak_idx]),
        iter_to_90pct_peak_gain=_iter_to_fraction(arr, 0.90),
        iter_to_95pct_peak_gain=_iter_to_fraction(arr, 0.95),
        diverged_after_peak=bool((arr[-1] - arr[peak_idx]) < -1e-3),
    )


def _parse_fortran_out(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for line in path.read_text().splitlines():
        match = ITER_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "iter": int(match.group("iter")),
                "lrate": float(match.group("lrate")),
                "ll": float(match.group("ll")),
                "nd": float(match.group("nd")),
                "seconds": float(match.group("seconds")),
            }
        )
    return pd.DataFrame(rows)


def _parse_fortran_total_hours(path: Path) -> float | None:
    for line in reversed(path.read_text().splitlines()):
        match = FORTRAN_TOTAL_HOURS_RE.search(line)
        if match:
            return float(match.group("hours"))
    return None


def _find_slurm_log(slurm_dir: Path, prefix: str, dataset_name: str) -> Path | None:
    matches = sorted(slurm_dir.glob(f"slurm-{prefix}_{dataset_name}-*.out"))
    if not matches:
        return None
    return matches[-1]


def _parse_python_total_seconds(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        match = PYTHON_TOTAL_SECONDS_RE.search(clean)
        if match:
            return float(match.group("seconds"))
    return None


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
    python_slurm_log = _find_slurm_log(python_dir.parent / "SLURM", "micaP", dataset_name)
    python_total_seconds = _parse_python_total_seconds(python_slurm_log)
    fortran_results = load_fortran_results(
        fortran_dir / "fortran_out",
        n_components=int(python_manifest["n_components"]),
        n_mixtures=int(python_manifest["n_mixtures"]),
        n_features=int(python_manifest["data_shape_samples_features"][1]),
    )
    fortran_ll = _clean_ll(fortran_results["LL"])
    fortran_iter_df = _parse_fortran_out(fortran_dir / "fortran_out" / "out.txt")
    fortran_total_hours = _parse_fortran_total_hours(fortran_dir / "fortran_out" / "out.txt")
    fortran_total_seconds = None if fortran_total_hours is None else 3600.0 * fortran_total_hours

    py_curve = _curve_metrics(python_ll)
    ft_curve = _curve_metrics(fortran_ll)

    shared_n = int(min(fortran_ll.size, python_ll.size))
    ll_delta = float(py_curve.ll_final - ft_curve.ll_final)
    shared_delta = python_ll[:shared_n] - fortran_ll[:shared_n]

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
        "fortran_iter_peak": ft_curve.iter_peak,
        "python_iter_peak": py_curve.iter_peak,
        "fortran_final_minus_peak": ft_curve.final_minus_peak,
        "python_final_minus_peak": py_curve.final_minus_peak,
        "fortran_diverged_after_peak": ft_curve.diverged_after_peak,
        "python_diverged_after_peak": py_curve.diverged_after_peak,
        "fortran_iter_to_90pct_peak_gain": ft_curve.iter_to_90pct_peak_gain,
        "python_iter_to_90pct_peak_gain": py_curve.iter_to_90pct_peak_gain,
        "fortran_iter_to_95pct_peak_gain": ft_curve.iter_to_95pct_peak_gain,
        "python_iter_to_95pct_peak_gain": py_curve.iter_to_95pct_peak_gain,
        "shared_n_iter": shared_n,
        "shared_ll_mae": float(np.mean(np.abs(shared_delta))),
        "shared_ll_rmse": float(np.sqrt(np.mean(shared_delta**2))),
        "shared_ll_max_abs_delta": float(np.max(np.abs(shared_delta))),
        "fortran_mean_seconds_per_iter": (
            float(fortran_iter_df["seconds"].mean()) if not fortran_iter_df.empty else None
        ),
        "fortran_total_hours_approx": fortran_total_hours,
        "fortran_total_seconds_approx": fortran_total_seconds,
        "python_total_seconds": python_total_seconds,
        "python_vs_fortran_time_ratio": (
            None
            if python_total_seconds is None or fortran_total_seconds is None or fortran_total_seconds <= 0
            else float(python_total_seconds / fortran_total_seconds)
        ),
        "fortran_vs_python_speedup": (
            None
            if python_total_seconds is None or fortran_total_seconds is None or python_total_seconds <= 0
            else float(fortran_total_seconds / python_total_seconds)
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
    record["python_slurm_log"] = None if python_slurm_log is None else str(python_slurm_log)
    return record


def _plot_ll_parity(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["fortran_final_ll"], df["python_final_ll"], s=50, color="#1f77b4")
    for row in df.itertuples(index=False):
        ax.annotate(row.dataset, (row.fortran_final_ll, row.python_final_ll), xytext=(4, 4), textcoords="offset points")
    finite = np.r_[df["fortran_final_ll"].to_numpy(), df["python_final_ll"].to_numpy()]
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="0.5")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Fortran final LL")
    ax.set_ylabel("Python final LL")
    ax.set_title("Final Log-Likelihood Parity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ll_delta(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.loc[df["dataset"] != "gv84"].sort_values("python_minus_fortran_ll")
    colors = np.where(
        plot_df["abs_python_minus_fortran_ll"] > 1e-2,
        "#d62728",
        "#2ca02c",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["dataset"], plot_df["python_minus_fortran_ll"], color=colors)
    ax.axhline(0.0, color="0.3", linewidth=1)
    ax.set_ylabel("Python - Fortran final LL")
    ax.set_title("Per-Dataset Final LL Difference")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_runtime_comparison(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.dropna(subset=["python_total_seconds", "fortran_total_seconds_approx"]).copy()
    plot_df = plot_df.loc[plot_df["dataset"] != "gv84"].sort_values("fortran_vs_python_speedup")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_df))
    width = 0.42
    ax.bar(x - width / 2, plot_df["fortran_total_seconds_approx"], width=width, label="Fortran (~sec)", color="#1f77b4")
    ax.bar(x + width / 2, plot_df["python_total_seconds"], width=width, label="Python (sec)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["dataset"], rotation=45)
    ax.set_ylabel("Approx wall time (seconds)")
    ax.set_title("Approximate Runtime Comparison (gv84 excluded)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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


def _make_markdown(df: pd.DataFrame, aggregate: dict[str, object]) -> str:
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
    if "fortran_total_seconds_approx" in df.columns and "python_total_seconds" in df.columns:
        display_cols.extend(["fortran_total_seconds_approx", "python_total_seconds", "fortran_vs_python_speedup"])

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
            f"- Median approximate Fortran/Python speed ratio: {aggregate['median_fortran_vs_python_speedup']:.3f}x"
        )
    lines.extend(["", table, ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize paired AMICA Python vs Fortran benchmark batches."
    )
    parser.add_argument("--fortran-batch-dir", type=Path, required=True)
    parser.add_argument("--python-batch-dir", type=Path, required=True)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-source-corr", action="store_true")
    args = parser.parse_args()

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
        "datasets_with_post_peak_divergence": sorted(
            df.loc[
                df["fortran_diverged_after_peak"] | df["python_diverged_after_peak"],
                "dataset",
            ].tolist()
        ),
        "median_python_iter_fraction_of_fortran": float(df["python_iter_fraction_of_fortran"].median()),
    }
    if "matched_source_abs_corr_mean" in df.columns:
        aggregate["mean_matched_source_abs_corr"] = float(df["matched_source_abs_corr_mean"].mean())
        aggregate["min_matched_source_abs_corr"] = float(df["matched_source_abs_corr_min"].min())
    timing_df = df.dropna(subset=["python_total_seconds", "fortran_total_seconds_approx"])
    if not timing_df.empty:
        aggregate["median_python_total_seconds"] = float(timing_df["python_total_seconds"].median())
        aggregate["median_fortran_total_seconds_approx"] = float(timing_df["fortran_total_seconds_approx"].median())
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
