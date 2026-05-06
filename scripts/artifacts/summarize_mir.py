#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DATASET_BY_NUM = {
    1: "km81",
    2: "jo74",
    3: "ds76",
    4: "cj82",
    5: "ap82",
    6: "ke70",
    7: "tp62",
    8: "cz84",
    9: "gm84",
    10: "gv84",
    11: "nf68",
    12: "ds80",
    13: "kb77",
    14: "ts79",
}


def mat_cellstr(values: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(values).ravel()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize MICA MIR results excluding MATLAB dataset 10 / gv84."
    )
    parser.add_argument(
        "--mir-mat",
        type=Path,
        default=Path(
            "/Users/scotterik/devel/projects/amica-python/amica-benchmark/"
            "benchmark_runs/mica_release_amica_python_matlab/mir_new.mat"
        ),
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path(
            "/Users/scotterik/devel/projects/amica-python/amica-benchmark/"
            "results/mir_summary"
        ),
    )
    parser.add_argument("--exclude-dataset-num", type=int, default=10)
    args = parser.parse_args()

    mat = loadmat(args.mir_mat.expanduser().resolve(), squeeze_me=True, struct_as_record=False)
    algorithms = mat_cellstr(mat["algorithms"])
    mir = np.asarray(mat["mir"], dtype=np.float64)
    keep_cols = [i for i in range(mir.shape[1]) if i != args.exclude_dataset_num - 1]

    rows = []
    for i, algorithm in enumerate(algorithms):
        values = mir[i, keep_cols]
        rows.append(
            {
                "algorithm": algorithm,
                "mean_mir": float(np.nanmean(values)),
                "std_mir": float(np.nanstd(values, ddof=1)),
                "n": int(np.sum(np.isfinite(values))),
            }
        )
    rows.sort(key=lambda row: row["mean_mir"], reverse=True)

    out_prefix = args.out_prefix.expanduser().resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "mean_mir", "std_mir", "n"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MIR Summary",
        "",
        f"Source: `{args.mir_mat.expanduser().resolve()}`",
        "",
        f"Excluded MATLAB dataset {args.exclude_dataset_num}: "
        f"`{DATASET_BY_NUM[args.exclude_dataset_num]}`.",
        "",
        "| Rank | Algorithm | Mean MIR | Std MIR | n |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['algorithm']} | {row['mean_mir']:.6f} | "
            f"{row['std_mir']:.6f} | {row['n']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
