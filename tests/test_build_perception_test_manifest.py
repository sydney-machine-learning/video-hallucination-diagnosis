from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_perception_test_manifest as builder


def make_annotations(count: int = 8):
    annotations = {}
    cuts = {}
    for index in range(count):
        video_id = f"video_{index}"
        annotations[video_id] = {
            "metadata": {
                "split": "valid",
                "video_id": video_id,
                "frame_rate": 10.0,
                "num_frames": 100,
            },
            "mc_question": [
                {
                    "id": 0,
                    "question": f"Question {index}?",
                    "options": ["one", "two", "three"],
                    "answer_id": index % 3,
                    "area": "physics",
                    "reasoning": "descriptive",
                    "tag": ["motion"],
                }
            ],
        }
        cuts[video_id] = -1
    return annotations, cuts


def test_selection_and_roles_ignore_answer_values():
    annotations, _ = make_annotations()
    selected = builder.select_video_ids(annotations, 6)
    roles = builder.assign_roles(selected, 3)
    changed = json.loads(json.dumps(annotations))
    for payload in changed.values():
        payload["mc_question"][0]["answer_id"] = 2
        payload["mc_question"][0]["question"] = "changed"
        payload["mc_question"][0]["options"] = ["x", "y", "z"]
    assert builder.select_video_ids(changed, 6) == selected
    assert builder.assign_roles(selected, 3) == roles


def test_official_frozen_manifest_has_no_labels():
    frozen = ROOT / "input" / "perception_test" / "frozen"
    manifest = frozen / "perception_test_valid500_inference.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1888
    assert len({row["video_id"] for row in rows}) == 500
    assert {row["role"] for row in rows} == {"calibration", "test"}
    assert all(not ({"answer", "answer_id", "gold"} & set(row)) for row in rows)
    assert all(len(row["candidates"]) == 3 for row in rows)
    assert all(row["end"] > row["start"] == 0.0 for row in rows)
    with (frozen / "pilot_labels.csv").open(encoding="utf-8", newline="") as handle:
        pilot = list(csv.DictReader(handle))
    with (frozen / "sealed_test_labels.csv").open(encoding="utf-8", newline="") as handle:
        test = list(csv.DictReader(handle))
    assert len(pilot) == 977
    assert len(test) == 911
    assert {row["question_id"] for row in pilot}.isdisjoint(
        row["question_id"] for row in test
    )


def test_perception_runner_record_is_label_free():
    katana = ROOT / "katana"
    if str(katana) not in sys.path:
        sys.path.insert(0, str(katana))
    import perception_multimodel_runner as perception

    manifest = ROOT / "input/perception_test/frozen/perception_test_valid500_inference.jsonl"
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    item = perception.runner.base.WorkItem(0, row, "base")
    record = perception.runner.base.base_record(
        item,
        "qwen",
        perception.runner.base.BACKENDS["qwen"]["model_id"],
        perception.runner.base.BACKENDS["qwen"]["revision"],
        "prompt",
        ["A", "B", "C"],
        Path(row["video_path"]),
        "0" * 64,
        list(range(16)),
        [float(index) for index in range(16)],
    )
    assert record["dataset"] == "Perception-Test-valid500"
    assert record["gold_candidate_index"] is None
    assert record["question_type"] in {"abstraction", "memory", "physics", "semantics"}
    assert "source_dataset" not in record
