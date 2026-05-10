from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import psutil


EPOCH_SLICE_BY_DATASET = {
    # gv84 contains a long noisy tail; match the clean segment used for AMICA reruns.
    "gv84": slice(None, 665),
}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _get_proc_status_value(name: str) -> str | None:
    status = _read_text(Path("/proc/self/status"))
    if status is None:
        return None
    prefix = f"{name}:"
    for line in status.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def _get_cpu_model() -> str | None:
    if shutil.which("lscpu") is not None:
        try:
            result = subprocess.run(
                ["lscpu"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Model name:"):
                    return line.split(":", 1)[1].strip()

    cpuinfo = _read_text(Path("/proc/cpuinfo"))
    if cpuinfo is None:
        return platform.processor() or None
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _get_cpu_affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _get_thread_cpu_allowed_lists() -> dict[str, str]:
    task_dir = Path("/proc/self/task")
    if not task_dir.exists():
        return {}
    allowed: dict[str, str] = {}
    for status_path in sorted(task_dir.glob("*/status")):
        status = _read_text(status_path)
        if status is None:
            continue
        for line in status.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                allowed[status_path.parent.name] = line.split(":", 1)[1].strip()
                break
    return allowed


def _get_thread_env() -> dict[str, str | None]:
    return {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OMP_PROC_BIND",
            "OMP_PLACES",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }


def _get_slurm_env() -> dict[str, str | None]:
    return {
        name: os.environ.get(name)
        for name in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_NODELIST",
            "SLURM_NODELIST",
            "SLURMD_NODENAME",
            "SLURM_NODEID",
            "SLURM_NTASKS",
            "SLURM_TASKS_PER_NODE",
            "SLURM_CPUS_PER_TASK",
            "SLURM_CPUS_ON_NODE",
            "SLURM_MEM_PER_NODE",
            "SLURM_MEM_PER_CPU",
            "SLURM_SUBMIT_HOST",
            "SLURM_SUBMIT_DIR",
            "SLURM_CLUSTER_NAME",
            "SLURM_PARTITION",
            "SLURM_JOB_ACCOUNT",
            "SLURM_JOB_QOS",
        )
    }


def collect_environment_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "sys_platform": sys.platform,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "node_name": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "cpu_model": _get_cpu_model(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_affinity": _get_cpu_affinity(),
        "cpus_allowed_list": _get_proc_status_value("Cpus_allowed_list"),
        "thread_cpus_allowed_list": _get_thread_cpu_allowed_lists(),
        "mems_allowed_list": _get_proc_status_value("Mems_allowed_list"),
        "memory_total_bytes": psutil.virtual_memory().total,
        "pid": os.getpid(),
        "os_cpu_count": os.cpu_count(),
        "thread_env": _get_thread_env(),
        "slurm": _get_slurm_env(),
    }
    try:
        import torch
    except ImportError:
        pass
    else:
        metadata["torch_num_threads"] = torch.get_num_threads()
        metadata["torch_num_interop_threads"] = torch.get_num_interop_threads()
    return metadata


def clean_ll(values: np.ndarray) -> np.ndarray:
    """Return finite nonzero log-likelihood values from an AMICA LL curve."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return arr[arr != 0]


def count_ll_iterations(values: np.ndarray) -> int:
    """Count completed AMICA iterations represented in an LL curve."""
    return int(clean_ll(values).size)


def load_eeglab_set(set_path: Path) -> np.ndarray:
    """Load EEGLAB .set as (n_samples, n_features) float64."""
    import mne

    set_path = Path(set_path).expanduser().resolve()
    if not set_path.exists():
        raise FileNotFoundError(f"Dataset .set file not found: {set_path}")

    # Most mica_release files are epoched, but we support raw .set as fallback.
    try:
        epochs = mne.read_epochs_eeglab(set_path, verbose="ERROR")
        epoch_slice = EPOCH_SLICE_BY_DATASET.get(set_path.stem)
        if epoch_slice is not None:
            epochs = epochs[epoch_slice]
        x3 = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_times)
        data = np.transpose(x3, (0, 2, 1)).reshape(-1, x3.shape[1])
    except Exception:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        data = raw.get_data().T
    return data.astype(np.float64, copy=False)


def marginal_entropies_getent2(u: np.ndarray, nbins: int | None = None) -> np.ndarray:
    """Estimate row-wise differential entropies using Delorme's getent2.m rule."""
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {u.shape}.")

    n_rows, n_samples = u.shape
    if nbins is None:
        nbins = min(100, int(round(np.sqrt(n_samples))))
    if nbins < 1:
        raise ValueError(f"nbins must be >= 1, got {nbins}.")

    entropies = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        row = u[i]
        row_min = float(np.min(row))
        row_max = float(np.max(row))
        delta = (row_max - row_min) / nbins
        if not np.isfinite(delta) or delta <= 0:
            entropies[i] = float("-inf")
            continue

        bins = 1 + np.round((nbins - 1) * (row - row_min) / (row_max - row_min))
        _, counts = np.unique(bins, return_counts=True)
        pmf = counts.astype(np.float64) / float(n_samples)
        entropies[i] = -float(np.sum(pmf * np.log(pmf))) + float(np.log(delta))
    return entropies


def mutual_information_reduction(
    data: np.ndarray,
    unmixing: np.ndarray,
    nbins: int | None = None,
) -> dict[str, object]:
    """Compute Delorme-style mutual-information reduction for a square unmixing."""
    data = np.asarray(data, dtype=np.float64)
    unmixing = np.asarray(unmixing, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected data as (samples, features), got {data.shape}.")
    if unmixing.ndim != 2 or unmixing.shape[0] != unmixing.shape[1]:
        raise ValueError(f"MIR requires a square unmixing matrix, got {unmixing.shape}.")
    if unmixing.shape[1] != data.shape[1]:
        raise ValueError(
            "Unmixing/data shape mismatch: "
            f"unmixing={unmixing.shape}, data={data.shape}."
        )

    h_data = marginal_entropies_getent2(data.T, nbins=nbins)
    sources = unmixing @ data.T
    h_sources = marginal_entropies_getent2(sources, nbins=nbins)
    sign, logabsdet = np.linalg.slogdet(unmixing)
    mir = float(np.sum(h_data) - np.sum(h_sources) + logabsdet)

    return {
        "mi_reduction": mir,
        "sum_data_entropy": float(np.sum(h_data)),
        "sum_source_entropy": float(np.sum(h_sources)),
        "logabsdet_unmixing": float(logabsdet),
        "det_sign": float(sign),
        "nbins": int(nbins) if nbins is not None else min(100, int(round(np.sqrt(data.shape[0])))),
        "data_entropies": h_data.tolist(),
        "source_entropies": h_sources.tolist(),
    }


def infer_fortran_n_components(fortran_out: Path) -> int:
    """Infer n_components from Fortran W file length."""
    w_path = Path(fortran_out).expanduser().resolve() / "W"
    if not w_path.exists():
        raise FileNotFoundError(f"Fortran W file not found: {w_path}")
    flat = np.fromfile(w_path, dtype=np.float64)
    n = int(round(np.sqrt(flat.size)))
    if n * n != flat.size:
        raise ValueError(
            f"Cannot infer square W shape from {w_path}: got {flat.size} values."
        )
    return n


def discover_fortran_out(dataset_set: Path, search_root: Path) -> Path:
    """Find best Fortran output dir for a dataset stem under a search root.

    Preference order:
    1) fortran_out under a folder exactly named like dataset stem
    2) fortran_out under a folder containing the stem
    3) newest modified W file
    """
    dataset_set = Path(dataset_set).expanduser().resolve()
    search_root = Path(search_root).expanduser().resolve()
    if not search_root.exists():
        raise FileNotFoundError(f"fortran search root does not exist: {search_root}")

    dataset_stem = dataset_set.stem
    candidates: list[tuple[int, float, Path]] = []

    for out_dir in search_root.rglob("fortran_out"):
        if not out_dir.is_dir():
            continue
        w_path = out_dir / "W"
        if not w_path.exists():
            continue

        parent_names = [p.name for p in out_dir.parents]
        if dataset_stem in parent_names:
            score = 2
        elif any(dataset_stem in name for name in parent_names):
            score = 1
        else:
            score = 0

        mtime = w_path.stat().st_mtime
        candidates.append((score, mtime, out_dir.resolve()))

    if not candidates:
        raise FileNotFoundError(
            f"No fortran_out directories with W found under: {search_root}"
        )

    # Highest score first, newest first.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates[0][2]

    if candidates[0][0] == 0:
        raise FileNotFoundError(
            "No dataset-matching Fortran output found. "
            f"Searched dataset stem '{dataset_stem}' under {search_root}."
        )

    return best
