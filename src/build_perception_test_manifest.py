"""Build a label-free, video-grouped Perception Test confirmation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "trace-rc-perception-test-valid500-manifest-1.0"
DATASET_REPOSITORY = "google-deepmind/perception_test"
DATASET_REVISION = "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"
ANNOTATION_URL = (
    "https://storage.googleapis.com/dm-perception-test/zip_data/"
    "mc_question_valid_annotations.zip"
)
VIDEO_ARCHIVE_URL = (
    "https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip"
)
SELECTION_SALT = "trace-rc-perception-test-valid500-selection-20260803"
SPLIT_SALT = "trace-rc-perception-test-valid500-role-split-20260803"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()


def select_video_ids(annotations: dict[str, Any], count: int) -> list[str]:
    """Select videos without inspecting questions, options, or answers."""
    if count <= 0 or count > len(annotations):
        raise ValueError("invalid selection size")
    return sorted(annotations, key=lambda value: (stable_key(SELECTION_SALT, value), value))[:count]


def assign_roles(video_ids: list[str], calibration_count: int) -> dict[str, str]:
    """Assign whole videos to roles independently of labels and question text."""
    if calibration_count <= 0 or calibration_count >= len(video_ids):
        raise ValueError("invalid calibration-video count")
    ordered = sorted(video_ids, key=lambda value: (stable_key(SPLIT_SALT, value), value))
    calibration = set(ordered[:calibration_count])
    return {
        video_id: ("calibration" if video_id in calibration else "test")
        for video_id in video_ids
    }


def write_labels(path: Path, rows: list[dict[str, Any]], role: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question_id", "video_id", "gold_candidate_index"],
        )
        writer.writeheader()
        for row in rows:
            if row["role"] == role:
                writer.writerow(
                    {
                        "question_id": row["question_id"],
                        "video_id": row["video_id"],
                        "gold_candidate_index": row["gold_candidate_index"],
                    }
                )


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    cuts = json.loads(args.cut_mapping.read_text(encoding="utf-8"))
    if not isinstance(annotations, dict) or len(annotations) != 5260:
        raise ValueError("unexpected official validation annotation inventory")
    selected = select_video_ids(annotations, args.video_count)
    roles = assign_roles(selected, args.calibration_videos)
    # The inference manifest is consumed on Katana.  Preserve the explicitly
    # supplied POSIX root instead of resolving it on the machine that builds
    # the seal (which may be Windows).
    video_root = PurePosixPath(args.video_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for video_id in sorted(selected):
        payload = annotations[video_id]
        metadata = payload["metadata"]
        if str(metadata["video_id"]) != video_id or metadata["split"] != "valid":
            raise ValueError(f"{video_id}: invalid metadata")
        fps = float(metadata["frame_rate"])
        frames = int(metadata["num_frames"])
        if fps <= 0 or frames <= 0:
            raise ValueError(f"{video_id}: invalid video timing")
        official_cut = int(cuts.get(video_id, -1))
        if official_cut == 0 or official_cut < -1 or official_cut > frames:
            raise ValueError(f"{video_id}: invalid official cut {official_cut}")
        end = (official_cut if official_cut > 0 else frames) / fps
        questions = payload.get("mc_question", [])
        if not questions:
            raise ValueError(f"{video_id}: no multiple-choice questions")
        seen_local_ids: set[int] = set()
        for question in questions:
            local_id = int(question["id"])
            if local_id in seen_local_ids:
                raise ValueError(f"{video_id}: duplicate local question id")
            seen_local_ids.add(local_id)
            choices = [str(value).strip() for value in question["options"]]
            if len(choices) != 3 or any(not value for value in choices):
                raise ValueError(f"{video_id}:{local_id}: invalid options")
            gold = int(question["answer_id"])
            if gold not in range(len(choices)):
                raise ValueError(f"{video_id}:{local_id}: invalid answer")
            question_id = f"perception-valid:{video_id}:{local_id}"
            role = roles[video_id]
            row = {
                "schema_version": SCHEMA_VERSION,
                "dataset_repository": DATASET_REPOSITORY,
                "dataset_revision": DATASET_REVISION,
                "split": "valid500",
                "source_dataset": "Perception Test validation",
                "role": role,
                "question_id": question_id,
                "video_id": video_id,
                "video_path": str(video_root / f"{video_id}.mp4"),
                "task": str(question["area"]),
                "reasoning": str(question["reasoning"]),
                "tags": [str(value) for value in question.get("tag", [])],
                "question": str(question["question"]).strip(),
                "candidates": choices,
                "start": 0.0,
                "end": float(end),
                "official_cut_frame": official_cut,
                "frame_rate": fps,
                "num_frames": frames,
            }
            manifest_rows.append(row)
            private_rows.append(
                {
                    "question_id": question_id,
                    "video_id": video_id,
                    "role": role,
                    "gold_candidate_index": gold,
                    "area": row["task"],
                    "reasoning": row["reasoning"],
                }
            )

    manifest_rows.sort(key=lambda row: row["question_id"])
    private_rows.sort(key=lambda row: row["question_id"])
    manifest_path = args.output_dir / "perception_test_valid500_inference.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in manifest_rows:
            if any(key in row for key in ("answer", "answer_id", "gold")):
                raise RuntimeError("label leaked into inference manifest")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    pilot_labels = args.output_dir / "pilot_labels.csv"
    test_labels = args.output_dir / "sealed_test_labels.csv"
    write_labels(pilot_labels, private_rows, "calibration")
    write_labels(test_labels, private_rows, "test")
    selected_ids_path = args.output_dir / "selected_video_ids.txt"
    with selected_ids_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(sorted(selected)) + "\n")

    by_role: dict[str, list[dict[str, Any]]] = {
        role: [row for row in private_rows if row["role"] == role]
        for role in ("calibration", "test")
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "annotation_url": ANNOTATION_URL,
        "video_archive_url": VIDEO_ARCHIVE_URL,
        "annotations_sha256": sha256_file(args.annotations),
        "cut_mapping_sha256": sha256_file(args.cut_mapping),
        "manifest_sha256": sha256_file(manifest_path),
        "pilot_labels_sha256": sha256_file(pilot_labels),
        "sealed_test_labels_sha256": sha256_file(test_labels),
        "selected_video_ids_sha256": sha256_file(selected_ids_path),
        "selection_salt": SELECTION_SALT,
        "split_salt": SPLIT_SALT,
        "selection_frame": "all 5,260 official validation videos with MCQA annotations",
        "questions": len(manifest_rows),
        "videos": len(selected),
        "calibration_questions": len(by_role["calibration"]),
        "test_questions": len(by_role["test"]),
        "calibration_videos": args.calibration_videos,
        "test_videos": args.video_count - args.calibration_videos,
        "area_distribution": distribution(private_rows, "area"),
        "reasoning_distribution": distribution(private_rows, "reasoning"),
        "role_area_distribution": {
            role: distribution(rows, "area") for role, rows in by_role.items()
        },
        "role_reasoning_distribution": {
            role: distribution(rows, "reasoning") for role, rows in by_role.items()
        },
        "selection_uses_question_text": False,
        "selection_uses_options": False,
        "selection_uses_answer_values": False,
        "role_assignment_uses_answer_values": False,
        "test_labels_opened_for_selection_or_split": False,
    }
    lock_path = args.output_dir / "preinference_split_lock.json"
    with lock_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cut-mapping", type=Path, required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-count", type=int, default=500)
    parser.add_argument("--calibration-videos", type=int, default=250)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
