#!/usr/bin/env python3
"""Download open EEG benchmark data and reference parameter files."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

FILES = {
    "eeglab_data.fdt": "https://github.com/sccn/eeglab/raw/develop/sample_data/eeglab_data.fdt",
    "eeglab_data.set": "https://github.com/sccn/eeglab/raw/develop/sample_data/eeglab_data.set",
    "amicadefs_test.param": "https://raw.githubusercontent.com/scott-huberty/amica/amica-python/tests/eeglab_sample_data/amicadefs_test.param",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "amica_test_data" / "eeglab_sample_data",
        help="Directory where benchmark dataset and param files are stored.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fname, url in FILES.items():
        dst = args.output_dir / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"exists: {dst}")
            continue
        print(f"downloading: {url}\n  -> {dst}")
        urlretrieve(url, dst)

    print(f"ready: {args.output_dir}")


if __name__ == "__main__":
    main()
