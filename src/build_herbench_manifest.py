"""Build the sealed HERBench Lite-v2 TRACE-RC split and label files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "trace-rc-herbench-lite-v2-manifest-1.0"
DATASET_REVISION = "DanBenAmi/HERBench@30b42e97180f2c7c0fdc3f2aed412ae04f0366ca"
SPLIT_SALT = "trace-rc-herbench-lite-v2-video-split-20260802"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}\0{value}".encode("utf-8")).hexdigest()


def strip_option_prefix(value: str, expected_index: int) -> str:
    label = chr(ord("A") + expected_index)
    text = str(value).strip()
    match = re.match(r"^([A-E])\.\s*(.*)$", text, flags=re.DOTALL)
    if match:
        if match.group(1) != label:
            raise ValueError(f"choice label mismatch: expected {label}, got {match.group(1)}")
        text = match.group(2).strip()
    if not text:
        raise ValueError("empty candidate text")
    return text


def normalize_choices(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(f"choices must be a sequence, got {type(value).__name__}")
    if len(values) not in {4, 5}:
        raise ValueError(f"HERBench requires four or five choices, got {len(values)}")
    return [strip_option_prefix(text, index) for index, text in enumerate(values)]


def assign_video_roles(frame: pd.DataFrame) -> dict[str, str]:
    videos = frame[["video_id", "source_dataset"]].drop_duplicates()
    if videos["video_id"].duplicated().any():
        raise ValueError("one video_id maps to multiple source datasets")
    pilot: set[str] = set()
    fractional: list[tuple[float, str, list[str]]] = []
    for source, group in videos.groupby("source_dataset", sort=True):
        identifiers = sorted(group["video_id"].astype(str), key=lambda x: (stable_key(x), x))
        exact = len(identifiers) / 2.0
        take = int(np.floor(exact))
        pilot.update(identifiers[:take])
        fractional.append((exact - take, str(source), identifiers[take:]))
    target = len(videos) // 2
    for _, _, remaining in sorted(fractional, key=lambda row: (-row[0], row[1])):
        if len(pilot) >= target:
            break
        if remaining:
            pilot.add(remaining[0])
    if len(videos) != 68 or len(pilot) != 34:
        raise ValueError(f"expected 68/34 videos, got {len(videos)}/{len(pilot)}")
    return {
        str(video_id): ("calibration" if str(video_id) in pilot else "test")
        for video_id in videos["video_id"]
    }


def write_labels(path: Path, rows: list[dict[str, Any]], role: str) -> None:
    selected = [row for row in rows if row["role"] == role]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question_id", "video_id", "gold_candidate_index"],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "video_id": row["video_id"],
                    "gold_candidate_index": row["gold_candidate_index"],
                }
            )


def build(args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.read_parquet(args.parquet)
    required = {
        "question_id",
        "video_id",
        "video_path",
        "question",
        "choices",
        "answer",
        "answer_index",
        "task_type",
        "source_dataset",
        "duration",
        "metadata_json",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing HERBench columns: {missing}")
    if len(frame) != 1971 or frame["video_id"].nunique() != 68:
        raise ValueError("unexpected HERBench Lite-v2 inventory")
    if frame["question_id"].duplicated().any():
        raise ValueError("duplicate question_id")
    roles = assign_video_roles(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_root = args.video_root.resolve()
    private_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for item in frame.sort_values("question_id", kind="stable").to_dict("records"):
        candidates = normalize_choices(item["choices"])
        answer_index = int(item["answer_index"])
        if answer_index not in range(len(candidates)):
            raise ValueError(f"{item['question_id']}: invalid answer index")
        expected_answer = chr(ord("A") + answer_index)
        if str(item["answer"]).strip() != expected_answer:
            raise ValueError(f"{item['question_id']}: answer/index disagreement")
        video_id = str(item["video_id"])
        video_path = (video_root / str(item["video_path"])).resolve()
        role = roles[video_id]
        manifest_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_repository": "DanBenAmi/HERBench",
                "dataset_revision": DATASET_REVISION,
                "split": "lite_v2",
                "role": role,
                "question_id": str(item["question_id"]),
                "video_id": video_id,
                "video_path": str(video_path),
                "task": str(item["task_type"]),
                "source_dataset": str(item["source_dataset"]),
                "question": str(item["question"]),
                "candidates": candidates,
                "start": 0.0,
                "end": float(item["duration"]),
                "metadata_json": str(item["metadata_json"]),
            }
        )
        private_rows.append(
            {
                "question_id": str(item["question_id"]),
                "video_id": video_id,
                "role": role,
                "gold_candidate_index": answer_index,
            }
        )
    manifest_path = args.output_dir / "herbench_lite_v2_inference.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in manifest_rows:
            if any(key in row for key in ("answer", "answer_index", "gold")):
                raise RuntimeError("label leaked into inference manifest")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    pilot_labels = args.output_dir / "pilot_labels.csv"
    test_labels = args.output_dir / "sealed_test_labels.csv"
    write_labels(pilot_labels, private_rows, "calibration")
    write_labels(test_labels, private_rows, "test")
    manifest_frame = pd.DataFrame(manifest_rows)
    source_role = (
        manifest_frame[["video_id", "source_dataset", "role"]]
        .drop_duplicates()
        .groupby(["source_dataset", "role"])
        .size()
        .unstack(fill_value=0)
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_revision": DATASET_REVISION,
        "parquet_sha256": sha256_file(args.parquet),
        "manifest_sha256": sha256_file(manifest_path),
        "pilot_labels_sha256": sha256_file(pilot_labels),
        "sealed_test_labels_sha256": sha256_file(test_labels),
        "split_salt": SPLIT_SALT,
        "questions": int(len(manifest_frame)),
        "videos": int(manifest_frame["video_id"].nunique()),
        "pilot_questions": int((manifest_frame["role"] == "calibration").sum()),
        "test_questions": int((manifest_frame["role"] == "test").sum()),
        "pilot_videos": int(manifest_frame.loc[manifest_frame["role"] == "calibration", "video_id"].nunique()),
        "test_videos": int(manifest_frame.loc[manifest_frame["role"] == "test", "video_id"].nunique()),
        "source_video_balance": {
            str(source): {str(role): int(value) for role, value in row.items()}
            for source, row in source_role.to_dict(orient="index").items()
        },
        "selection_uses_answer_values": False,
        "test_labels_opened_for_split": False,
    }
    lock_path = args.output_dir / "preinference_split_lock.json"
    lock_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
