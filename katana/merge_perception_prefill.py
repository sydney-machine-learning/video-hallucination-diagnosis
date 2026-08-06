#!/usr/bin/env python3
"""Atomically merge successful prefill records into a resumable output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_records(path: Path, *, allow_missing: bool) -> list[dict]:
    if not path.is_file():
        if allow_missing:
            return []
        raise FileNotFoundError(path)
    rows: list[dict] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("record_key", ""))
        if not key or key in seen:
            raise ValueError(f"missing/duplicate record_key at {path}:{line_number}")
        seen.add(key)
        rows.append(row)
    return rows


def merge(source: Path, destination: Path, expected_max: int) -> dict:
    existing = read_records(destination, allow_missing=True)
    incoming = read_records(source, allow_missing=False)
    by_key = {str(row["record_key"]): row for row in existing if row.get("status") == "ok"}
    accepted = 0
    skipped_non_ok = 0
    for row in incoming:
        if row.get("status") != "ok":
            skipped_non_ok += 1
            continue
        key = str(row["record_key"])
        if key not in by_key:
            by_key[key] = row
            accepted += 1
    if len(by_key) > expected_max:
        raise ValueError(f"merged record count {len(by_key)} exceeds {expected_max}")
    ordered = sorted(by_key.values(), key=lambda row: (int(row["ordinal"]), str(row["record_key"])))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".merge-{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "existing_ok": len(existing),
        "incoming": len(incoming),
        "accepted_new_ok": accepted,
        "skipped_non_ok": skipped_non_ok,
        "merged_ok": len(ordered),
        "expected_max": expected_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-max", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.source, args.destination, args.expected_max), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
