#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from export_amica_python_to_mica_mat import export_amica_python_to_mica_mat


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MICA_ROOT = Path("/Users/scotterik/amica_test_data/mica_release")
DEFAULT_PYTHON_RUN_ROOT = (
    REPO_ROOT / "benchmark_runs" / "mica_release_python_slurm_20260419_174859"
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
        "dipfit1.02",
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
    if "AMICA-Python" in text:
        return

    anchor = (
        "allalgs(end+1).algo = 'binica';        allalgs(end).speed = 0;     "
        "allalgs(end).name = 'binica ext.';    allalgs(end).options = { 'extended' 1 };\n"
    )
    insert = (
        "allalgs(end+1).algo = 'amica_python';  allalgs(end).speed = 0;     "
        "allalgs(end).name = 'AMICA-Python'; allalgs(end).options = {  };\n"
    )
    if anchor not in text:
        raise ValueError(f"Could not find processdat.m algorithm insertion anchor in {path}")
    text = text.replace(anchor, anchor + insert)
    text = text.replace("%allalgs(48).", "%allalgs(49).")
    path.write_text(text)


def patch_mutualinfoalgo(path: Path) -> None:
    text = path.read_text()
    if "'AMICA-Python'" not in text:
        text = text.replace(
            "algorithms = {  'Amica' 'Ext. Infomax' 'Pearson' 'Infomax' ...",
            "algorithms = {  'Amica' 'AMICA-Python' 'Ext. Infomax' 'Pearson' 'Infomax' ...",
        )

    skip_block = (
        "      if strcmp(algorithms{algo}, 'AMICA-Python') && dat == 10\n"
        "          h(:,algo,dat) = NaN;\n"
        "          mir(algo,dat) = NaN;\n"
        "          continue;\n"
        "      end\n"
    )
    if skip_block not in text:
        loop_anchor = "for algo=1:length(algorithms)\n   for dat=1:14\n"
        if loop_anchor not in text:
            raise ValueError(f"Could not find mutualinfoalgo.m loop anchor in {path}")
        text = text.replace(loop_anchor, loop_anchor + skip_block)

    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text)


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
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument(
        "--eeglab-dir",
        type=Path,
        default=REPO_ROOT / "matlab" / "eeglab11_0_3_1b",
        help="Validated for convenience; MATLAB command still needs this on path.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mica_root = args.mica_root.expanduser().resolve()
    python_run_root = args.python_run_root.expanduser().resolve()
    workdir = args.workdir.expanduser().resolve()
    validate_eeglab(args.eeglab_dir.expanduser().resolve())

    copy_or_link_mica_release(mica_root, workdir, force=args.force)
    patch_processdat(workdir / "processdat.m")
    patch_mutualinfoalgo(workdir / "mutualinfoalgo.m")
    export_result = export_amica_python_to_mica_mat(
        python_run_root=python_run_root,
        out_dir=workdir / "icadecompositions",
        excluded={"gv84"},
    )

    result = {
        "workdir": str(workdir),
        "mica_root": str(mica_root),
        "python_run_root": str(python_run_root),
        "eeglab_dir": str(args.eeglab_dir.expanduser().resolve()),
        **export_result,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
