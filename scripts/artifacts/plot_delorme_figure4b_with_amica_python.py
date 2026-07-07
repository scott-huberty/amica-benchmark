#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from scipy.stats import linregress


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


@dataclass(frozen=True)
class AlgorithmInfo:
    number: int
    name: str
    code: str


def add_cluster_regions(ax):
    from matplotlib.patches import Ellipse

    regions = [
        # center_x, center_y, width, height, color,
        (41.86, 3.8, 0.12, 3.4, "#71D083"),  # PCA; Grass
        (42.29, 8.0, 0.62, 13.5, "#E796F3"), # AMUSE through JADE-TD-ish; Plum
        (42.67, 17.6, 0.32, 6.8, "#FF92AD"), # JADE opt. through FastICA; Crimson
        (43.08, 27.7, 0.34, 8.2, "#FFCA16"), # Pearson through AMICA; Amber
    ]

    for x, y, width, height, color in regions:
        ax.add_patch(
            Ellipse(
                (x, y),
                width=width,
                height=height,
                facecolor=color,
                edgecolor="none",
                alpha=0.35,
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


def write_summary(
    rows: list[dict[str, object]],
    regression_original: dict[str, float],
    regression_augmented: dict[str, float],
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
        "## Coordinates",
        "",
        "| Rank by MIR | Algorithm | MIR (kbits/s) | RV < 5% components (%) | Count |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
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
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
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
            label=str(row["algorithm"]),
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
        ax.text(
            float(row["mean_mir_kbits_s"]) + dx,
            float(row["rv_below_5_percent"]) + dy,
            str(row["algorithm"]),
            fontsize=9,
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
    original_rows = [row for row in rows if row["algorithm"] not in PYTHON_ALGORITHMS]
    regression_original = regression(original_rows)
    regression_augmented = regression(rows)

    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    write_csv(rows, csv_path)
    write_summary(rows, regression_original, regression_augmented, md_path, workdir)
    if not args.no_amica_python:
        plot_rows(rows, regression_original, regression_augmented, png_path, pdf_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if not args.no_amica_python:
        print(f"Wrote {png_path}")
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
