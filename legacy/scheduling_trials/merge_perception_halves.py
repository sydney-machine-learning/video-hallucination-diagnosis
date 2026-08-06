#!/usr/bin/env python3
"""Merge two disjoint Perception Test inference halves with strict checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import perception_multimodel_runner as perception


def load(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = record.get("record_key")
        if not key or key in records:
            raise ValueError(f"missing or duplicate key at {path}:{number}")
        records[key] = record
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = perception.runner.base.load_manifest(
        args.manifest,
        roles={"calibration", "test"},
        views=tuple(perception.runner.base.VIEW_SPECS),
        max_questions=None,
    )
    expected = [item.key for item in sorted(items, key=lambda item: item.ordinal)]
    if len(expected) < 6000 or len(expected) != len(set(expected)):
        raise ValueError("unexpected sealed record-key inventory")
    merged: dict[str, dict] = {}
    for path in (args.head, args.tail):
        for key, record in load(path).items():
            if key in merged:
                raise ValueError(f"overlapping inference halves: {key}")
            merged[key] = record
    missing = [key for key in expected if key not in merged]
    extra = sorted(set(merged).difference(expected))
    errors = [key for key in expected if merged.get(key, {}).get("status") != "ok"]
    if missing or extra or errors:
        raise ValueError(
            f"invalid merge: missing={len(missing)} extra={len(extra)} errors={len(errors)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for key in expected:
            handle.write(json.dumps(merged[key], sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"TRACE_RC_PERCEPTION_MERGE_PASS records={len(expected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
