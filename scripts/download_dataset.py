#!/usr/bin/env python3
"""Download the optional EEGLAB MICA benchmark dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from amica.utils import fetch_mica_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "amica_test_data",
        help="Directory where the extracted mica_release benchmark data should live.",
    )
    args = parser.parse_args()

    release_dir = fetch_mica_release(args.output_dir)
    print(f"ready: {release_dir}")


if __name__ == "__main__":
    main()
