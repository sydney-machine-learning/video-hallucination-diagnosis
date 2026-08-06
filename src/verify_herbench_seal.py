#!/usr/bin/env python3
"""Verify that every file covered by the HERBench pre-inference seal is unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-root", type=Path, default=Path("."))
    parser.add_argument(
        "--seal",
        type=Path,
        default=Path("HERBENCH_PREINFERENCE_SEAL_20260802.json"),
    )
    args = parser.parse_args()

    root = args.route_root.resolve()
    seal_path = args.seal if args.seal.is_absolute() else root / args.seal
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    failures: list[dict[str, str]] = []

    for relative, expected in seal["source_hashes"].items():
        path = root / relative
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
            continue
        observed = sha256(path)
        if observed != expected:
            failures.append(
                {"path": relative, "expected": expected, "observed": observed}
            )

    extra_checks = {
        "input/herbench_lite_v2_inference.jsonl": seal["manifest_sha256"],
        "input/pilot_labels.csv": seal["pilot_labels_sha256"],
        "input/sealed_test_labels.csv": seal["sealed_test_labels_sha256"],
        seal["development_authorization"]["path"]: seal["development_authorization"][
            "sha256"
        ],
    }
    for relative, expected in extra_checks.items():
        path = root / relative
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
            continue
        observed = sha256(path)
        if observed != expected:
            failures.append(
                {"path": relative, "expected": expected, "observed": observed}
            )

    report = {
        "seal": str(seal_path),
        "sealed_source_count": len(seal["source_hashes"]),
        "extra_hash_count": len(extra_checks),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
