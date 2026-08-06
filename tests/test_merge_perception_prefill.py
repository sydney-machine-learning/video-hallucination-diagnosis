from __future__ import annotations

import json
import sys
from pathlib import Path


KATANA = Path(__file__).resolve().parents[1] / "katana"
sys.path.insert(0, str(KATANA))

from merge_perception_prefill import merge


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prefill_merge_is_atomic_deduplicated_and_success_only(tmp_path: Path) -> None:
    destination = tmp_path / "canonical.jsonl"
    source = tmp_path / "prefill.jsonl"
    write_rows(destination, [{"record_key": "b", "ordinal": 1, "status": "ok"}])
    write_rows(
        source,
        [
            {"record_key": "a", "ordinal": 0, "status": "ok"},
            {"record_key": "b", "ordinal": 1, "status": "ok", "new": True},
            {"record_key": "c", "ordinal": 2, "status": "error"},
        ],
    )
    report = merge(source, destination, 3)
    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert [row["record_key"] for row in rows] == ["a", "b"]
    assert "new" not in rows[1]
    assert report["accepted_new_ok"] == 1
    assert report["skipped_non_ok"] == 1
