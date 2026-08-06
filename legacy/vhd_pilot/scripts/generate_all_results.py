from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script), *args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all VHD result tables and figures.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()

    repeats = str(args.bootstrap_repeats)
    run(
        "make_main_results_figure.py",
        "--manifest",
        str(args.manifest.resolve()),
        "--bootstrap-repeats",
        repeats,
    )
    run("make_source_split_figures.py", "--bootstrap-repeats", repeats)
    run("make_category_heatmaps.py")


if __name__ == "__main__":
    main()
