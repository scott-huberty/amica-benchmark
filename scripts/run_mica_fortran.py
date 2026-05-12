#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from amica.utils.fortran import write_data, write_param_file

from mica_common import (
    collect_environment_metadata,
    collect_fortran_software_metadata,
    count_ll_iterations,
    load_eeglab_set,
)


DEFAULT_BENCH_ROOT = Path(__file__).resolve().parents[1] / "benchmark_runs"
DEFAULT_DATASETS_DIR = Path.home() / "amica_test_data" / "mica_release" / "datasets"


def _run_and_tee(cmd: list[str], log_path: Path) -> None:
    """Run a command, streaming combined stdout/stderr to both console and file."""
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
            return_code = process.wait()
        finally:
            process.stdout.close()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_fortran(
    *,
    dataset_set: Path,
    run_dir: Path,
    max_iter: int,
    n_components: int | None,
    n_mixtures: int,
    fortran_image: str,
    container_runtime: str = "docker",
    apptainer_image: str | None = None,
    fortran_threads: int = 1,
) -> dict[str, object]:
    dataset_set = dataset_set.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_set.stem
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    data = load_eeglab_set(dataset_set)

    data_fpath = dataset_dir / f"{dataset_name}_concat.fdt"
    write_data(data, data_fpath)

    fortran_out = run_dir / "fortran_out"
    fortran_out.mkdir(parents=True, exist_ok=True)

    param_path = run_dir / f"amicadefs_{dataset_name}.param"
    param_kwargs: dict[str, object] = {
        "files": f"/data/{data_fpath.name}",
        # NOTE: trailing slash is required by Fortran AMICA for tmp files.
        "outdir": "/out/fortran_out/",
        "max_iter": int(max_iter),
        "max_threads": int(fortran_threads),
        "num_mix_comps": int(n_mixtures),
    }
    if n_components is not None:
        param_kwargs["pcakeep"] = int(n_components)

    _, params = write_param_file(
        param_path,
        data=data,
        **param_kwargs,
    )

    if container_runtime == "docker":
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{dataset_dir}:/data",
            "-v",
            f"{run_dir}:/out",
            "-v",
            f"{run_dir}:/params",
            fortran_image,
            f"/params/{param_path.name}",
        ]
    else:
        image_ref = apptainer_image or f"docker://{fortran_image}"
        cmd = [
            "apptainer",
            "run",
            "--bind",
            f"{dataset_dir}:/data",
            "--bind",
            f"{run_dir}:/out",
            "--bind",
            f"{run_dir}:/params",
            image_ref,
            f"/params/{param_path.name}",
        ]
    log_path = run_dir / "fortran_console.txt"
    fit_started_at_epoch_seconds = time.time()
    fit_started_at_perf_counter = time.perf_counter()
    _run_and_tee(cmd, log_path)
    fit_finished_at_epoch_seconds = time.time()
    fit_seconds = time.perf_counter() - fit_started_at_perf_counter
    ll_path = fortran_out / "LL"
    fortran_n_iter = None
    if ll_path.exists():
        fortran_n_iter = count_ll_iterations(np.fromfile(ll_path, dtype=np.float64))

    manifest = {
        "dataset_set": str(dataset_set),
        "dataset_name": dataset_name,
        "data_shape_samples_features": [int(data.shape[0]), int(data.shape[1])],
        "block_size": int(params.block_size),
        "max_iter": int(max_iter),
        "n_components": None if n_components is None else int(n_components),
        "n_mixtures": int(n_mixtures),
        "run_dir": str(run_dir),
        "data_fdt": str(data_fpath),
        "fortran_param_file": str(param_path),
        "fortran_output_dir": str(fortran_out),
        "fortran_image": fortran_image,
        "container_runtime": container_runtime,
        "apptainer_image": apptainer_image,
        "software": {
            "fortran": collect_fortran_software_metadata(
                fortran_image=fortran_image,
                container_runtime=container_runtime,
                apptainer_image=apptainer_image,
            ),
        },
        "fortran_threads": int(fortran_threads),
        "environment": collect_environment_metadata(),
        "fit_started_at_epoch_seconds": float(fit_started_at_epoch_seconds),
        "fit_finished_at_epoch_seconds": float(fit_finished_at_epoch_seconds),
        "fit_seconds": float(fit_seconds),
        "fortran_n_iter": fortran_n_iter,
        "fortran_console_log": str(log_path),
    }
    manifest_path = run_dir / "fortran_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _default_run_dir(dataset_set: Path, bench_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return bench_root / f"mica_release_{dataset_set.stem}_{ts}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Fortran AMICA on one EEGLAB .set dataset via Docker or Apptainer."
    )
    parser.add_argument(
        "--dataset-set",
        type=Path,
        default=DEFAULT_DATASETS_DIR / "cz84.set",
        help="Path to EEGLAB .set file.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output run directory. Defaults to benchmark_runs/mica_release_<stem>_<timestamp>.",
    )
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="pcakeep for Fortran. Leave unset for default behavior.",
    )
    parser.add_argument("--n-mixtures", type=int, default=3)
    parser.add_argument(
        "--container-runtime",
        choices=("docker", "apptainer"),
        default="docker",
        help="Container runtime to use: docker (local) or apptainer (cluster).",
    )
    parser.add_argument("--fortran-image", default="shuberty/amica:latest")
    parser.add_argument(
        "--apptainer-image",
        default=None,
        help=(
            "Apptainer image URI/path (e.g., docker://shuberty/amica:latest or .sif). "
            "If unset, defaults to docker://<fortran-image>."
        ),
    )
    parser.add_argument("--fortran-threads", type=int, default=1)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir is not None else _default_run_dir(args.dataset_set, args.bench_root)
    manifest = run_fortran(
        dataset_set=args.dataset_set,
        run_dir=run_dir,
        max_iter=args.max_iter,
        n_components=args.n_components,
        n_mixtures=args.n_mixtures,
        fortran_image=args.fortran_image,
        container_runtime=args.container_runtime,
        apptainer_image=args.apptainer_image,
        fortran_threads=args.fortran_threads,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
