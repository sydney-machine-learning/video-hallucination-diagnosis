"""Verify every frozen Perception Test code and data digest before execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    root = args.route_root.resolve()
    seal = json.loads(args.seal.read_text(encoding="utf-8"))
    for relative, expected in seal["source_hashes"].items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"frozen digest mismatch: {relative}")
    frozen = root / "input/perception_test/frozen"
    direct = {
        "manifest_sha256": frozen / "perception_test_valid500_inference.jsonl",
        "pilot_labels_sha256": frozen / "pilot_labels.csv",
        "sealed_test_labels_sha256": frozen / "sealed_test_labels.csv",
        "selected_video_ids_sha256": frozen / "selected_video_ids.txt",
    }
    for key, path in direct.items():
        if digest(path) != seal[key]:
            raise RuntimeError(f"sealed data digest mismatch: {key}")
    rows = [
        json.loads(line)
        for line in direct["manifest_sha256"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1888 or len({row["video_id"] for row in rows}) != 500:
        raise RuntimeError("sealed manifest inventory mismatch")
    if any({"answer", "answer_id", "gold"} & set(row) for row in rows):
        raise RuntimeError("label leaked into sealed inference manifest")
    print(
        "TRACE_RC_PERCEPTION_SEAL_PASS "
        f"files={len(seal['source_hashes'])} questions={len(rows)} videos=500"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
