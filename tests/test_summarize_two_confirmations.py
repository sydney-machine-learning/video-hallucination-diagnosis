from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import summarize_two_confirmations as summary


SOURCE = (
    ROOT
    / "results/herbench_confirmation/8918264.kman.restech.unsw.edu.au"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_archive(path: Path, target: str) -> Path:
    shutil.copytree(SOURCE, path)
    gate_path = path / "confirmation_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["target_dataset"] = target
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [item for item in path.iterdir() if item.is_file() and item.name != "archive.sha256"]
    with (path / "archive.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for item in sorted(files):
            handle.write(f"{digest(item)}  /sealed/{target}/{item.name}\n")
    return path


def test_two_confirmation_summary_is_non_pooled(tmp_path: Path):
    first = make_archive(tmp_path / "first", "HERBench-Lite-v2")
    second = make_archive(tmp_path / "second", "Perception-Test-valid500")
    output = tmp_path / "output"
    argv = sys.argv
    try:
        sys.argv = [
            "summarize_two_confirmations.py",
            "--herbench-dir",
            str(first),
            "--perception-dir",
            str(second),
            "--output-dir",
            str(output),
        ]
        assert summary.main() == 0
    finally:
        sys.argv = argv
    payload = json.loads((output / "two_confirmation_summary.json").read_text(encoding="utf-8"))
    assert payload["all_confirmation_gates_pass"] is True
    assert payload["aggregation_policy"].startswith("report benchmark-specific")
    assert "pooled" not in payload
    assert payload["total_final_test_videos_descriptive"] == 68
    assert (output / "trace_rc_two_confirmation_table.tex").is_file()


def test_duplicate_target_is_rejected(tmp_path: Path):
    first = make_archive(tmp_path / "first", "HERBench-Lite-v2")
    second = make_archive(tmp_path / "second", "HERBench-Lite-v2")
    argv = sys.argv
    try:
        sys.argv = [
            "summarize_two_confirmations.py",
            "--herbench-dir",
            str(first),
            "--perception-dir",
            str(second),
            "--output-dir",
            str(tmp_path / "output"),
        ]
        with pytest.raises(RuntimeError, match="same target dataset"):
            summary.main()
    finally:
        sys.argv = argv


def test_archive_tampering_is_rejected(tmp_path: Path):
    archive = make_archive(tmp_path / "archive", "HERBench-Lite-v2")
    with (archive / "test_model_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(RuntimeError, match="archive digest mismatch"):
        summary.load_archive(archive, "tampered")
