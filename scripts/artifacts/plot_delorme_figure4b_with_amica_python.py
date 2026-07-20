#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import linregress, ttest_rel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKDIR = REPO_ROOT / "benchmark_runs" / "mica_release_amica_python_matlab"
DEFAULT_OUT_PREFIX = REPO_ROOT / "results" / "delorme_figure4b_with_amica_python"
DATASET_RANGE = tuple([*range(1, 10), *range(11, 15)])
N_COMPONENTS = 71
MIR_SCALE_TO_KBITS_PER_SEC = 1.4427 * 250.0 / 1000.0


FIGURE4B_ALGORITHMS = [
    "Amica",
    "Ext. Infomax",
    "Pearson",
    "Infomax",
    "SHIBBS",
    "FastICA",
    "JADE",
    "TICA",
    "JADE opt.",
    "JADE-TD",
    "FOBI",
    "SOBIRO",
    "EVD24",
    "EVD",
    "SOBI",
    "icaMS",
    "AMUSE",
    "PCA",
]
PYTHON_ALGORITHMS = ("Py-EM", "Py-DAAREM")
EXPECTED_ALGORITHM_NUMBERS = {
    "Infomax": 1,
    "Ext. Infomax": 2,
    "PCA": 43,
    "Amica": 45,
    "Py-EM": 48,
    "Py-DAAREM": 49,
}
ALGO_LABEL = {"Py-EM": "AMICA-Python", "Py-DAAREM": "AMICA-Python (DAAREM)"}

@dataclass(frozen=True)
class AlgorithmInfo:
    number: int
    name: str
    code: str


def add_cluster_regions(ax):
    from matplotlib.patches import Ellipse

    regions = [
        # center_x, center_y, width, height, color,
        (41.86, 3.8, 0.12, 3.4, "#EBFECF"),  # PCA; Grass
        (42.29, 8.0, 0.62, 13.5, "#DFE3F3"), # AMUSE through JADE-TD-ish; Plum
        (42.67, 17.6, 0.32, 6.8, "#FAE4F4"), # JADE opt. through FastICA; Crimson
        (43.08, 27.7, 0.34, 8.2, "#FFFFD1"), # Pearson through AMICA; Amber
    ]

    for x, y, width, height, color in regions:
        ax.add_patch(
            Ellipse(
                (x, y),
                width=width,
                height=height,
                facecolor=color,
                edgecolor="none",
                alpha=0.85,
                zorder=1,
            )
        )

def mat_cellstr(values: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(values).ravel()]


def parse_processdat_algorithms(path: Path) -> dict[str, AlgorithmInfo]:
    algorithms: list[AlgorithmInfo] = []
    current_code: str | None = None
    current_name: str | None = None
    current_number: int | None = None

    algo_pattern = re.compile(r"allalgs\(\s*(end\+1|\d+)\s*\)\.algo\s*=\s*'([^']*)'")
    name_pattern = re.compile(r"allalgs\(\s*end\s*\)\.name\s*=\s*'([^']*)'")

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("%"):
            continue
        algo_match = algo_pattern.search(line)
        name_match = name_pattern.search(line)
        if algo_match:
            index_expr = algo_match.group(1)
            current_number = (
                len(algorithms) + 1 if index_expr == "end+1" else int(index_expr)
            )
            current_code = algo_match.group(2)
            current_name = None
        if name_match:
            current_name = name_match.group(1)
        if (
            current_number is not None
            and current_code is not None
            and current_name is not None
        ):
            algorithms.append(
                AlgorithmInfo(
                    number=current_number,
                    name=current_name,
                    code=current_code,
                )
            )
            current_number = None
            current_code = None
            current_name = None

    return {algorithm.name: algorithm for algorithm in algorithms}


def load_mir_by_algorithm(mir_mat: Path) -> tuple[list[str], np.ndarray]:
    mat = loadmat(mir_mat, squeeze_me=True, struct_as_record=False)
    return mat_cellstr(mat["algorithms"]), np.asarray(mat["mir"], dtype=np.float64)


def load_algorithm_rvs(workdir: Path, algorithm: AlgorithmInfo) -> np.ndarray:
    values = []
    for dataset in DATASET_RANGE:
        mat_path = (
            workdir
            / "icadecompositions"
            / f"ica{dataset}_72_{algorithm.number:02d}_{algorithm.code}.mat"
        )
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing decomposition file: {mat_path}")
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "allrv" not in mat:
            raise KeyError(f"{mat_path} does not contain allrv")
        rv = np.asarray(mat["allrv"], dtype=np.float64).reshape(-1)
        if rv.size != N_COMPONENTS:
            raise ValueError(f"{mat_path} has {rv.size} allrv values, expected {N_COMPONENTS}")
        values.append(rv)
    return np.concatenate(values)


def load_algorithm_rvs_by_dataset(
    workdir: Path, algorithm: AlgorithmInfo
) -> dict[int, np.ndarray]:
    values = {}
    for dataset in DATASET_RANGE:
        mat_path = (
            workdir
            / "icadecompositions"
            / f"ica{dataset}_72_{algorithm.number:02d}_{algorithm.code}.mat"
        )
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing decomposition file: {mat_path}")
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "allrv" not in mat:
            raise KeyError(f"{mat_path} does not contain allrv")
        rv = np.asarray(mat["allrv"], dtype=np.float64).reshape(-1)
        if rv.size != N_COMPONENTS:
            raise ValueError(f"{mat_path} has {rv.size} allrv values, expected {N_COMPONENTS}")
        values[dataset] = rv
    return values


def build_rows(workdir: Path, include_amica_python: bool) -> list[dict[str, object]]:
    processdat_algorithms = parse_processdat_algorithms(workdir / "processdat.m")
    for algorithm, expected_number in EXPECTED_ALGORITHM_NUMBERS.items():
        if algorithm not in processdat_algorithms:
            raise KeyError(f"{algorithm!r} not found in {workdir / 'processdat.m'}")
        actual_number = processdat_algorithms[algorithm].number
        if actual_number != expected_number:
            raise ValueError(
                f"{algorithm!r} parsed as algorithm {actual_number}, expected "
                f"{expected_number}. Check processdat.m parsing before plotting."
            )

    mir_algorithms, mir = load_mir_by_algorithm(workdir / "mir_new.mat")
    keep_columns = [dataset - 1 for dataset in DATASET_RANGE]

    algorithms = list(FIGURE4B_ALGORITHMS)
    if include_amica_python:
        for algorithm in reversed(PYTHON_ALGORITHMS):
            algorithms.insert(1, algorithm)

    rows: list[dict[str, object]] = []
    for algorithm in algorithms:
        if algorithm not in processdat_algorithms:
            raise KeyError(f"{algorithm!r} not found in {workdir / 'processdat.m'}")
        if algorithm not in mir_algorithms:
            raise KeyError(f"{algorithm!r} not found in {workdir / 'mir_new.mat'}")

        mir_index = mir_algorithms.index(algorithm)
        mir_values = mir[mir_index, keep_columns]
        rv_values = load_algorithm_rvs(workdir, processdat_algorithms[algorithm])
        finite_rv_count = int(np.sum(np.isfinite(rv_values)))
        if algorithm in PYTHON_ALGORITHMS and finite_rv_count != rv_values.size:
            print(
                f"{algorithm} has {finite_rv_count}/{rv_values.size} finite allrv values. "
                "Non-finite RV values are treated as not near-dipolar.",
                file=sys.stderr,
            )
        rv_below_5 = int(np.sum(rv_values < 0.05))
        n_expected = len(DATASET_RANGE) * N_COMPONENTS
        rows.append(
            {
                "algorithm": algorithm,
                "mean_mir_raw": float(np.nanmean(mir_values)),
                "mean_mir_kbits_s": float(np.nanmean(mir_values) * MIR_SCALE_TO_KBITS_PER_SEC),
                "rv_below_5_count": rv_below_5,
                "rv_below_5_percent": float(rv_below_5 / n_expected * 100.0),
                "finite_rv_count": finite_rv_count,
                "n_components_expected": n_expected,
                "n_datasets": len(DATASET_RANGE),
            }
        )
    return rows


def metric_prefix(method: str) -> str:
    return {
        "Amica": "fortran_amica",
        "Py-EM": "amica_python",
        "Py-DAAREM": "amica_python_daarem",
    }[method]


def build_per_recording_rows(workdir: Path) -> list[dict[str, object]]:
    processdat_algorithms = parse_processdat_algorithms(workdir / "processdat.m")
    mir_algorithms, mir = load_mir_by_algorithm(workdir / "mir_new.mat")
    methods = ("Amica", *PYTHON_ALGORITHMS)

    rv_by_method = {
        method: load_algorithm_rvs_by_dataset(workdir, processdat_algorithms[method])
        for method in methods
    }
    rows: list[dict[str, object]] = []
    for dataset in DATASET_RANGE:
        row: dict[str, object] = {"dataset": dataset}
        mir_col = dataset - 1
        for method in methods:
            if method not in mir_algorithms:
                raise KeyError(f"{method!r} not found in {workdir / 'mir_new.mat'}")
            prefix = metric_prefix(method)
            mir_index = mir_algorithms.index(method)
            rv_values = rv_by_method[method][dataset]
            row[f"{prefix}_mir_kbits_s"] = float(
                mir[mir_index, mir_col] * MIR_SCALE_TO_KBITS_PER_SEC
            )
            row[f"{prefix}_rv_below_5_percent"] = float(
                np.sum(rv_values < 0.05) / N_COMPONENTS * 100.0
            )
            row[f"{prefix}_rv_below_5_count"] = int(np.sum(rv_values < 0.05))
            row[f"{prefix}_finite_rv_count"] = int(np.sum(np.isfinite(rv_values)))
        rows.append(row)
    return rows


def mean_sd(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
    }


def summarize_per_recording(rows: list[dict[str, object]]) -> dict[str, object]:
    method_labels = {
        "fortran_amica": "Fortran AMICA",
        "amica_python": "AMICA-Python",
        "amica_python_daarem": "AMICA-Python (DAAREM)",
    }
    metrics = ("mir_kbits_s", "rv_below_5_percent")
    summary: dict[str, object] = {
        "n_recordings": len(rows),
        "datasets": [int(row["dataset"]) for row in rows],
        "methods": {},
        "paired_tests": {},
    }

    for prefix, label in method_labels.items():
        metric_summary = {}
        for metric in metrics:
            values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows])
            metric_summary[metric] = mean_sd(values)
        summary["methods"][prefix] = {"label": label, **metric_summary}

    comparisons = {
        "amica_python_vs_fortran": ("amica_python", "fortran_amica"),
        "amica_python_daarem_vs_fortran": ("amica_python_daarem", "fortran_amica"),
        "amica_python_daarem_vs_amica_python": (
            "amica_python_daarem",
            "amica_python",
        ),
    }
    for comparison, (left, right) in comparisons.items():
        comparison_summary = {}
        for metric in metrics:
            left_values = np.asarray([float(row[f"{left}_{metric}"]) for row in rows])
            right_values = np.asarray([float(row[f"{right}_{metric}"]) for row in rows])
            diff = left_values - right_values
            test = ttest_rel(left_values, right_values)
            comparison_summary[metric] = {
                "mean_difference": float(np.mean(diff)),
                "sd_difference": float(np.std(diff, ddof=1)),
                "t_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "df": len(rows) - 1,
            }
        summary["paired_tests"][comparison] = comparison_summary

    return summary


def regression(rows: list[dict[str, object]]) -> dict[str, float]:
    x = np.asarray([row["mean_mir_kbits_s"] for row in rows], dtype=np.float64)
    y = np.asarray([row["rv_below_5_percent"] for row in rows], dtype=np.float64)
    result = linregress(x, y)
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_squared": float(result.rvalue**2),
        "p_value": float(result.pvalue),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "algorithm",
        "mean_mir_raw",
        "mean_mir_kbits_s",
        "rv_below_5_count",
        "rv_below_5_percent",
        "finite_rv_count",
        "n_components_expected",
        "n_datasets",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_per_recording_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "fortran_amica_mir_kbits_s",
        "amica_python_mir_kbits_s",
        "amica_python_daarem_mir_kbits_s",
        "fortran_amica_rv_below_5_percent",
        "amica_python_rv_below_5_percent",
        "amica_python_daarem_rv_below_5_percent",
        "fortran_amica_rv_below_5_count",
        "amica_python_rv_below_5_count",
        "amica_python_daarem_rv_below_5_count",
        "fortran_amica_finite_rv_count",
        "amica_python_finite_rv_count",
        "amica_python_daarem_finite_rv_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    rows: list[dict[str, object]],
    regression_original: dict[str, float],
    regression_augmented: dict[str, float],
    per_recording_summary: dict[str, object],
    path: Path,
    workdir: Path,
) -> None:
    ranked = sorted(rows, key=lambda row: row["mean_mir_kbits_s"], reverse=True)
    lines = [
        "# Delorme Figure 4B Data With Py-EM and Py-DAAREM",
        "",
        f"Source workdir: `{workdir}`",
        "",
        "Reduction mirrors `plotresults.m`: datasets `[1:9, 11:14]`, "
        "`mir * 1.4427 * 250 / 1000`, and `count(allrv < 0.05) / (13 * 71) * 100`.",
        "",
        "## Regression",
        "",
        "| Point set | R^2 | p-value | slope | intercept |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Original 18 algorithms | {regression_original['r_squared']:.6f} | "
            f"{regression_original['p_value']:.6e} | {regression_original['slope']:.6f} | "
            f"{regression_original['intercept']:.6f} |"
        ),
        (
            f"| Original + Python algorithms | {regression_augmented['r_squared']:.6f} | "
            f"{regression_augmented['p_value']:.6e} | {regression_augmented['slope']:.6f} | "
            f"{regression_augmented['intercept']:.6f} |"
        ),
        "",
        "## Per-recording AMICA comparisons",
        "",
        "| Method | MIR mean +/- SD (kbits/s) | RV < 5% mean +/- SD (%) |",
        "| --- | ---: | ---: |",
    ]

    methods = per_recording_summary["methods"]
    for key in ("fortran_amica", "amica_python", "amica_python_daarem"):
        method = methods[key]
        mir = method["mir_kbits_s"]
        rv = method["rv_below_5_percent"]
        lines.append(
            f"| {method['label']} | {mir['mean']:.3f} +/- {mir['sd']:.3f} | "
            f"{rv['mean']:.2f} +/- {rv['sd']:.2f} |"
        )

    lines.extend([
        "",
        "| Paired comparison | Metric | Mean difference | t(df) | p-value |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    comparison_labels = {
        "amica_python_vs_fortran": "AMICA-Python - Fortran AMICA",
        "amica_python_daarem_vs_fortran": "AMICA-Python (DAAREM) - Fortran AMICA",
        "amica_python_daarem_vs_amica_python": (
            "AMICA-Python (DAAREM) - AMICA-Python"
        ),
    }
    metric_labels = {
        "mir_kbits_s": "MIR (kbits/s)",
        "rv_below_5_percent": "RV < 5% (%)",
    }
    for comparison_key, comparison_label in comparison_labels.items():
        comparison = per_recording_summary["paired_tests"][comparison_key]
        for metric_key, metric_label in metric_labels.items():
            metric = comparison[metric_key]
            lines.append(
                f"| {comparison_label} | {metric_label} | "
                f"{metric['mean_difference']:.3f} | "
                f"{metric['t_statistic']:.2f}({metric['df']}) | "
                f"{metric['p_value']:.4f} |"
            )

    lines.extend([
        "",
        "## Coordinates",
        "",
        "| Rank by MIR | Algorithm | MIR (kbits/s) | RV < 5% components (%) | Count |",
        "| ---: | --- | ---: | ---: | ---: |",
    ])
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['algorithm']} | {row['mean_mir_kbits_s']:.6f} | "
            f"{row['rv_below_5_percent']:.6f} | {row['rv_below_5_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_rows(
    rows: list[dict[str, object]],
    regression_original: dict[str, float],
    regression_augmented: dict[str, float],
    png_path: Path,
    pdf_path: Path,
) -> None:
    original_rows = [row for row in rows if row["algorithm"] not in PYTHON_ALGORITHMS]
    python_rows = [row for row in rows if row["algorithm"] in PYTHON_ALGORITHMS]

    with plt.rc_context({
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }):
        fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)

    x_original = np.asarray([row["mean_mir_kbits_s"] for row in original_rows], dtype=float)
    y_original = np.asarray([row["rv_below_5_percent"] for row in original_rows], dtype=float)
    ax.scatter(x_original, y_original, color="#111111", s=36, zorder=3, alpha=0.75)

    python_styles = {
        "Py-EM": {"color": "#5472E4", "marker": "D"},
        "Py-DAAREM": {"color": "#D66A2C", "marker": "^"},
    }
    for row in python_rows:
        style = python_styles[str(row["algorithm"])]
        ax.scatter(
            [row["mean_mir_kbits_s"]],
            [row["rv_below_5_percent"]],
            color=style["color"],
            marker=style["marker"],
            s=68,
            zorder=4,
            label=str(ALGO_LABEL.get(row["algorithm"], row["algorithm"])),
        )

    for row in rows:
        dx = -.25 if row["algorithm"] in PYTHON_ALGORITHMS else 0.015
        dy = 0.0
        color = (
            python_styles[str(row["algorithm"])]["color"]
            if row["algorithm"] in PYTHON_ALGORITHMS
            else "#111111"
        )
        if row["algorithm"] in {"Amica", *PYTHON_ALGORITHMS}:
            dy = 0.4
        # Try to get annotation labels to be in position similar to orginal figure
        LABEL_OFFSETS = {
            "PCA": (-0.03, 1),
            "AMUSE": (0.025, 0),
            "icaMS": (0.03, 0),
            "EVD": (-0.1, 0),
            "EVD24": (-0.15, .25),
            "FOBI": (0.02, -0.6),
            "SOBIRO": (-0.17, 0.5),
            "JADE-TD": (0.03, 0),
            "SOBI": (-0.03, -1.5),
            "JADE opt.": (0.03, 0),
            "TICA": (0.02, -0.9),
            "JADE": (0.025, -0.7),
            "SHIBBS": (0.025, 0),
            "FastICA": (-0.08, 1),
            "Pearson": (-0.18, 0),
            "Ext. Infomax": (0, -1.2),
            "Infomax": (0.03, 0),
            "Amica": (0.02, 0),
            "Py-EM": (-0.32, 0),
            "Py-DAAREM": (-0.53, 0),
        }
        dx, dy = LABEL_OFFSETS[row["algorithm"]]
        x_text = float(row["mean_mir_kbits_s"]) + dx
        y_text = float(row["rv_below_5_percent"]) + dy
        ax.text(
            x_text,
            y_text,
            str(ALGO_LABEL.get(row["algorithm"], row["algorithm"])),
            fontsize=12,
            color=color,
        )

    x_min = min(row["mean_mir_kbits_s"] for row in rows)
    x_max = max(row["mean_mir_kbits_s"] for row in rows)
    pad = (x_max - x_min) * 0.12
    x_line = np.linspace(x_min - pad, x_max + pad, 200)
    y_original_line = (
        regression_original["intercept"] + regression_original["slope"] * x_line
    )
    y_augmented_line = (
        regression_augmented["intercept"] + regression_augmented["slope"] * x_line
    )
    ax.plot(x_line, y_original_line, "k--", linewidth=1.5, label="Original regression")
    ax.plot(
        x_line,
        y_augmented_line,
        color="#6E56CF",
        linestyle=":",
        linewidth=1.5,
        label="With Python algorithms",
    )

    add_cluster_regions(ax)
    ax.set_xlabel("Mutual information reduction (kbits/s)")
    ax.set_ylabel("Near-dipolar components, r.v. < 5% (%)")
    ax.set_xlim(x_min - pad, x_max + pad)
    y_max = max(row["rv_below_5_percent"] for row in rows)
    ax.set_ylim(0, max(35.0, y_max + 4.0))
    ax.text(
        0.63,
        0.17,
        (
            f"Original R^2={regression_original['r_squared']:.2f}\n"
            f"With Python R^2={regression_augmented['r_squared']:.2f}"
        ),
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#EDEDED", linewidth=0.8)
    ax.set_axisbelow(True)
    
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate Delorme Figure 4B coordinates and add Py-EM/Py-DAAREM."
    )
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument(
        "--no-amica-python",
        action="store_true",
        help="Only emit the original 18 Delorme algorithms.",
    )
    args = parser.parse_args()

    workdir = args.workdir.expanduser().resolve()
    out_prefix = args.out_prefix.expanduser().resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(workdir, include_amica_python=not args.no_amica_python)
    per_recording_rows = build_per_recording_rows(workdir)
    per_recording_summary = summarize_per_recording(per_recording_rows)
    original_rows = [row for row in rows if row["algorithm"] not in PYTHON_ALGORITHMS]
    regression_original = regression(original_rows)
    regression_augmented = regression(rows)

    csv_path = out_prefix.with_suffix(".csv")
    per_recording_csv_path = out_prefix.with_name(
        f"{out_prefix.name}_per_recording.csv"
    )
    per_recording_stats_path = out_prefix.with_name(
        f"{out_prefix.name}_per_recording_stats.json"
    )
    md_path = out_prefix.with_suffix(".md")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    write_csv(rows, csv_path)
    write_per_recording_csv(per_recording_rows, per_recording_csv_path)
    per_recording_stats_path.write_text(
        json.dumps(per_recording_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(
        rows,
        regression_original,
        regression_augmented,
        per_recording_summary,
        md_path,
        workdir,
    )
    if not args.no_amica_python:
        plot_rows(rows, regression_original, regression_augmented, png_path, pdf_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {per_recording_csv_path}")
    print(f"Wrote {per_recording_stats_path}")
    print(f"Wrote {md_path}")
    if not args.no_amica_python:
        print(f"Wrote {png_path}")
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
