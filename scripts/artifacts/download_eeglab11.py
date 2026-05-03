#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


EEGLAB_URL = "https://sccn.ucsd.edu/eeglab/download/daily/eeglab11_0_3_1b.zip"
DEFAULT_DEST = (
    Path(__file__).resolve().parents[2] / "matlab" / "eeglab11_0_3_1b"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EEGLAB 11.0.3.1b.")
    parser.add_argument("--url", default=EEGLAB_URL)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dest = args.dest.expanduser().resolve()
    if dest.exists():
        if not args.force:
            raise FileExistsError(f"{dest} exists. Re-run with --force to replace it.")
        shutil.rmtree(dest)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        zip_path = tmpdir / "eeglab11_0_3_1b.zip"
        print(f"Downloading {args.url}")
        urllib.request.urlretrieve(args.url, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)

        extracted = tmpdir / "eeglab11_0_3_1b"
        if not (extracted / "eeglab.m").exists():
            candidates = [p for p in tmpdir.iterdir() if (p / "eeglab.m").exists()]
            if len(candidates) != 1:
                raise FileNotFoundError("Could not locate extracted eeglab.m")
            extracted = candidates[0]

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), dest)
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
