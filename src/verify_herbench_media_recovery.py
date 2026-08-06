#!/usr/bin/env python3
"""Verify that every sealed HERBench manifest path resolves after extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paths = {Path(row["video_path"]) for row in rows}
    missing = sorted(str(path) for path in paths if not path.is_file())
    if len(rows) != 1971 or len(paths) != 68 or missing:
        raise SystemExit(
            f"media recovery failed rows={len(rows)} videos={len(paths)} "
            f"missing={len(missing)} first={missing[:5]}"
        )
    print(
        "HERBENCH_MEDIA_RECOVERY_PASS",
        len(rows),
        len(paths),
        sum(path.stat().st_size for path in paths),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
