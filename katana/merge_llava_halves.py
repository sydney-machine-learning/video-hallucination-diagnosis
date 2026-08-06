#!/usr/bin/env python3
"""Merge two label-free LLaVA recovery halves with strict key validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import herbench_multimodel_runner as her


def load_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = record.get("record_key")
        if not key or key in records:
            raise ValueError(f"missing or duplicate record_key at {path}:{number}")
        records[key] = record
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items = her.base.load_manifest(
        args.manifest,
        roles={"calibration", "test"},
        views=tuple(her.base.VIEW_SPECS),
        max_questions=None,
    )
    expected_order = [item.key for item in sorted(items, key=lambda item: item.ordinal)]
    if len(expected_order) != 7884 or len(set(expected_order)) != 7884:
        raise ValueError("sealed manifest does not produce 7884 unique record keys")

    merged: dict[str, dict] = {}
    for path in (args.head, args.tail):
        for key, record in load_records(path).items():
            if key in merged and merged[key] != record:
                raise ValueError(f"non-identical duplicate record: {key}")
            merged[key] = record
    missing = [key for key in expected_order if key not in merged]
    extra = sorted(set(merged).difference(expected_order))
    errors = [key for key in expected_order if merged.get(key, {}).get("status") != "ok"]
    if missing or extra or errors:
        raise ValueError(
            f"invalid merge: missing={len(missing)} extra={len(extra)} errors={len(errors)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for key in expected_order:
            handle.write(json.dumps(merged[key], sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"TRACE_RC_LLAVA_MERGE_PASS records={len(expected_order)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
