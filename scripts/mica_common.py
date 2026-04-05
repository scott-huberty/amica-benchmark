from __future__ import annotations

from pathlib import Path

import numpy as np


def load_eeglab_set(set_path: Path) -> np.ndarray:
    """Load EEGLAB .set as (n_samples, n_features) float64."""
    import mne

    set_path = Path(set_path).expanduser().resolve()
    if not set_path.exists():
        raise FileNotFoundError(f"Dataset .set file not found: {set_path}")

    # Most mica_release files are epoched, but we support raw .set as fallback.
    try:
        epochs = mne.read_epochs_eeglab(set_path, verbose="ERROR")
        x3 = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_times)
        data = np.transpose(x3, (0, 2, 1)).reshape(-1, x3.shape[1])
    except Exception:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        data = raw.get_data().T
    return data.astype(np.float64, copy=False)


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
