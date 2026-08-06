"""Verify selected Perception Test media and write an atomic prep marker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    paths = {str(row["video_id"]): Path(row["video_path"]) for row in rows}
    if len(rows) != 1888 or len(paths) != 500 or inventory["videos"] != 500:
        raise RuntimeError("Perception Test media inventory count mismatch")
    records = {str(record["video_id"]): record for record in inventory["records"]}
    if set(records) != set(paths):
        raise RuntimeError("manifest and media inventory video IDs differ")
    for index, video_id in enumerate(sorted(paths), 1):
        path = paths[video_id]
        record = records[video_id]
        if not path.is_file() or path.stat().st_size != int(record["size"]):
            raise RuntimeError(f"missing or short media: {video_id}")
        if digest(path) != record["sha256"]:
            raise RuntimeError(f"media digest mismatch: {video_id}")
        if index % 50 == 0:
            print(f"verified {index}/500 videos", flush=True)
    payload = {
        "schema_version": "trace-rc-perception-test-valid500-prep-complete-1.0",
        "videos": len(paths),
        "questions": len(rows),
        "total_bytes": inventory["total_bytes"],
        "manifest_sha256": digest(args.manifest),
        "inventory_sha256": digest(args.inventory),
        "all_media_size_and_sha256_checks_pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
