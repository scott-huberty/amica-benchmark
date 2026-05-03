#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.io import savemat


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


def full_unmixing(model) -> np.ndarray:
    components = getattr(model, "components_", None)
    if components is not None:
        return np.asarray(components, dtype=np.float64)

    unmixing = np.asarray(model._unmixing, dtype=np.float64)
    whitening = np.asarray(model.whitening_, dtype=np.float64)
    return unmixing @ whitening


def export_amica_python_to_mica_mat(
    *,
    python_run_root: Path,
    out_dir: Path,
    algorithm_num: int = 48,
    algorithm_slug: str = "amica_python",
    excluded: set[str] | None = None,
) -> dict[str, list[str]]:
    excluded = excluded or {"gv84"}
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped: list[str] = []
    for dataset_num, dataset in DATASET_BY_NUM.items():
        if dataset in excluded:
            skipped.append(dataset)
            continue

        model_path = python_run_root / dataset / "python_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        model = joblib.load(model_path)
        w = full_unmixing(model)
        if w.shape != (71, 71):
            raise ValueError(f"{dataset}: expected W shape (71, 71), got {w.shape}")

        manifest_path = python_run_root / dataset / "python_run.json"
        timeelapsed = np.nan
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            timeelapsed = float(manifest.get("fit_seconds", np.nan))

        out_path = out_dir / (
            f"ica{dataset_num}_72_{algorithm_num:02d}_{algorithm_slug}.mat"
        )
        savemat(
            out_path,
            {
                "W": w,
                "timeelapsed": np.asarray([[timeelapsed]], dtype=np.float64),
                "allrv": np.full((1, 71), np.nan, dtype=np.float64),
                "allposxyz": np.full((3, 71), np.nan, dtype=np.float64),
                "allmomxyz": np.full((3, 71), np.nan, dtype=np.float64),
            },
            format="5",
        )
        written.append(str(out_path))

    return {"written": written, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export saved AMICA-Python joblib models as MICA decomposition .mat files."
    )
    parser.add_argument(
        "--python-run-root",
        type=Path,
        default=Path(
            "/Users/scotterik/devel/projects/amica-python/amica-benchmark/"
            "benchmark_runs/mica_release_python_slurm_20260419_174859"
        ),
    )
    parser.add_argument(
        "--mica-root",
        type=Path,
        default=Path("/Users/scotterik/amica_test_data/mica_release"),
        help="Used only to default --out-dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for exported .mat files. Defaults to <mica-root>/icadecompositions.",
    )
    parser.add_argument("--algorithm-num", type=int, default=48)
    parser.add_argument("--algorithm-slug", default="amica_python")
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=["gv84"],
        help="Dataset stem to skip. Repeatable. Defaults to gv84.",
    )
    args = parser.parse_args()

    mica_root = args.mica_root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else mica_root / "icadecompositions"
    )
    result = export_amica_python_to_mica_mat(
        python_run_root=args.python_run_root.expanduser().resolve(),
        out_dir=out_dir,
        algorithm_num=args.algorithm_num,
        algorithm_slug=args.algorithm_slug,
        excluded=set(args.exclude_dataset or []),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
