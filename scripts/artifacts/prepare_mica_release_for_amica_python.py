#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from export_amica_python_to_mica_mat import export_amica_python_to_mica_mat


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(__file__).resolve().parent
DEFAULT_MICA_ROOT = Path("/Users/scotterik/amica_test_data/mica_release")
DEFAULT_PYTHON_RUN_ROOT = (
    REPO_ROOT / "benchmark_runs" / "mica_release_python_slurm_20260419_174859"
)
DEFAULT_TRIPLET_RUN_DIR = (
    REPO_ROOT / "benchmark_runs" / "mica_release_all_run-1_20260703_115448"
)
DEFAULT_WORKDIR = REPO_ROOT / "benchmark_runs" / "mica_release_amica_python_matlab"


def copy_or_link_mica_release(source: Path, dest: Path, *, force: bool) -> None:
    if dest.exists():
        if not force:
            raise FileExistsError(f"{dest} exists. Re-run with --force to replace it.")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    link_names = {
        "datasets",
        "icadecompositions",
        "randdip",
        "pmi_save.mat",
    }
    for item in source.iterdir():
        target = dest / item.name
        if item.name in link_names:
            target.symlink_to(item)
        elif item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    decompositions = dest / "icadecompositions"
    if decompositions.is_symlink():
        decompositions.unlink()
        shutil.copytree(source / "icadecompositions", decompositions)


def patch_processdat(path: Path) -> None:
    text = path.read_text()
    if "Py-EM" in text and "Py-DAAREM" in text:
        return

    anchor = (
        "allalgs(end+1).algo = 'binica';        allalgs(end).speed = 0;     "
        "allalgs(end).name = 'binica ext.';    allalgs(end).options = { 'extended' 1 };\n"
    )
    insert = (
        "allalgs(end+1).algo = 'amica_python_em';      allalgs(end).speed = 0;     "
        "allalgs(end).name = 'Py-EM'; allalgs(end).options = {  };\n"
        "allalgs(end+1).algo = 'amica_python_daarem';  allalgs(end).speed = 0;     "
        "allalgs(end).name = 'Py-DAAREM'; allalgs(end).options = {  };\n"
    )
    if anchor not in text:
        raise ValueError(f"Could not find processdat.m algorithm insertion anchor in {path}")
    text = text.replace(anchor, anchor + insert)
    text = text.replace("%allalgs(48).", "%allalgs(50).")
    path.write_text(text)


def patch_mutualinfoalgo(path: Path) -> None:
    text = path.read_text()
    if "'Py-EM'" not in text or "'Py-DAAREM'" not in text:
        text = text.replace(
            "algorithms = {  'Amica' 'Ext. Infomax' 'Pearson' 'Infomax' ...",
            "algorithms = {  'Amica' 'Py-EM' 'Py-DAAREM' 'Ext. Infomax' 'Pearson' 'Infomax' ...",
        )

    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text)


def patch_plgndr(path: Path) -> None:
    path.write_text(
        """function [y] = plgndr(n,k,x)
% PLGNDR associated Legendre function.
%
% The original DIPFIT 1.02 distribution shipped this as an architecture-
% specific MEX file. Modern macOS MATLAB cannot load the bundled mexmac
% binary, so this prepared working copy uses MATLAB's built-in LEGENDRE.

if nargin ~= 3
    error('invalid number of arguments for PLGNDR');
end
if k < 0 || k > n || x > 1.0 || x < -1.0
    error('Bad arguments in routine plgndr');
end

p = legendre(n, x);
y = p(k+1);
"""
    )


def copy_artifact_matlab_scripts(workdir: Path) -> None:
    matlab_dir = ARTIFACTS_DIR / "matlab"
    for script in matlab_dir.glob("*.m"):
        shutil.copy2(script, workdir / script.name)


def validate_eeglab(eeglab_dir: Path) -> None:
    if not (eeglab_dir / "eeglab.m").exists():
        raise FileNotFoundError(
            f"EEGLAB not found at {eeglab_dir}. "
            "Run scripts/artifacts/download_eeglab11.py or set --eeglab-dir."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a patched MICA release workdir with AMICA-Python decompositions."
    )
    parser.add_argument("--mica-root", type=Path, default=DEFAULT_MICA_ROOT)
    parser.add_argument("--python-run-root", type=Path, default=DEFAULT_PYTHON_RUN_ROOT)
    parser.add_argument(
        "--triplet-run-dir",
        type=Path,
        default=DEFAULT_TRIPLET_RUN_DIR,
        help=(
            "Canonical triplet run directory containing <dataset>/python_em and "
            "<dataset>/python_daarem. Used unless --legacy-single-python is set."
        ),
    )
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument(
        "--eeglab-dir",
        type=Path,
        default=REPO_ROOT / "matlab" / "eeglab11_0_3_1b",
        help="Validated for convenience; MATLAB command still needs this on path.",
    )
    parser.add_argument(
        "--legacy-single-python",
        action="store_true",
        help="Export one legacy AMICA-Python algorithm from --python-run-root.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mica_root = args.mica_root.expanduser().resolve()
    python_run_root = args.python_run_root.expanduser().resolve()
    triplet_run_dir = args.triplet_run_dir.expanduser().resolve()
    workdir = args.workdir.expanduser().resolve()
    validate_eeglab(args.eeglab_dir.expanduser().resolve())

    copy_or_link_mica_release(mica_root, workdir, force=args.force)
    patch_processdat(workdir / "processdat.m")
    patch_mutualinfoalgo(workdir / "mutualinfoalgo.m")
    patch_plgndr(workdir / "dipfit1.02" / "copyprivate" / "plgndr.m")
    copy_artifact_matlab_scripts(workdir)
    if args.legacy_single_python:
        export_result = {
            "legacy_amica_python": export_amica_python_to_mica_mat(
                python_run_root=python_run_root,
                out_dir=workdir / "icadecompositions",
            )
        }
    else:
        export_result = {
            "py_em": export_amica_python_to_mica_mat(
                python_run_root=triplet_run_dir,
                python_subdir="python_em",
                out_dir=workdir / "icadecompositions",
                algorithm_num=48,
                algorithm_slug="amica_python_em",
            ),
            "py_daarem": export_amica_python_to_mica_mat(
                python_run_root=triplet_run_dir,
                python_subdir="python_daarem",
                out_dir=workdir / "icadecompositions",
                algorithm_num=49,
                algorithm_slug="amica_python_daarem",
            ),
        }

    result = {
        "workdir": str(workdir),
        "mica_root": str(mica_root),
        "python_run_root": str(python_run_root),
        "triplet_run_dir": str(triplet_run_dir),
        "eeglab_dir": str(args.eeglab_dir.expanduser().resolve()),
        "exports": export_result,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
