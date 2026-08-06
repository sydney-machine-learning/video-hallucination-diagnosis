#!/usr/bin/env python3
"""Three-model, four-view HERBench Lite-v2 TRACE-RC runner."""

from __future__ import annotations

import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CAVER_KATANA = Path.home() / "caver_route" / "katana"
SICR_KATANA = Path.home() / "sicr_route" / "katana"
WORKSPACE = Path(__file__).resolve().parents[2]
for candidate in (
    HERE,
    CAVER_KATANA,
    SICR_KATANA,
    WORKSPACE / "caver_route" / "katana",
    WORKSPACE / "sicr_route" / "katana",
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import mvbench_multimodel_runner as base  # noqa: E402
import nextgqa_internvl_runner as internvl_runtime  # noqa: E402
import star_multimodel_runner as star  # noqa: E402


base.SCHEMA_VERSION = "trace-rc-herbench-lite-v2-fresh-inference-1.0"
base.SEED = 20260802
base.VIEW_SPECS = {
    "base": ("reference", "uniform_full_video_16_frames", 16),
    "invariant_nuisance": ("invariant", "half_step_phase_shift_8_frames", 8),
    "dense_acquisition": ("evidence_addition", "uniform_full_video_24_frames", 24),
    "temporal_order_destruction": ("evidence_removal", "reverse_uniform_8_frames", 8),
}
base.BACKENDS = {
    "qwen": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    },
    "internvl": {
        "model_id": "OpenGVLab/InternVL3_5-14B-Instruct",
        "revision": "72c82460a2c05a3bc2b47be230c44edc88e86ed4",
    },
    "llava": {
        "model_id": "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        "revision": "802f16c8346a062fef7ddf92b2bd64a770a50a1a",
    },
}
# ``mvbench_multimodel_runner`` re-exports ``load_adapter`` from the STAR
# runner.  That function resolves its validation table in the defining module,
# so update both module namespaces before any adapter is constructed.
star.BACKENDS = base.BACKENDS
# The reusable InternVL loader also carries a revision guard from its original
# 4B job.  The outer adapter has already checked the HERBench sealed table;
# point the inner guard at that same pinned 14B identifier and revision.
internvl_runtime.MODEL_ID = base.BACKENDS["internvl"]["model_id"]
internvl_runtime.MODEL_REVISION = base.BACKENDS["internvl"]["revision"]


_base_load_manifest = base.load_manifest


def load_manifest(path, roles, views, max_questions):
    """Keep each question's views together while grouping equal video segments.

    The inherited runner retains decoded frames only for the immediately
    preceding video segment.  HERBench interleaves questions from the same
    video, so processing manifest order would repeatedly decode that video.
    Sorting the in-memory work queue changes only execution order: original
    ordinals, record keys, prompts, views, and the sealed question set remain
    unchanged.
    """
    items = _base_load_manifest(path, roles, views, max_questions)
    view_rank = {view_id: rank for rank, view_id in enumerate(views)}

    def segment_key(item):
        question = item.question
        start = question.get("start")
        end = question.get("end")
        return (
            str(question["video_path"]),
            "" if start is None else str(start),
            "" if end is None else str(end),
            str(question["question_id"]),
            view_rank[item.view_id],
        )

    return sorted(items, key=segment_key)


base.load_manifest = load_manifest


def sample_frame_indices(
    frame_count: int,
    duration_seconds: float,
    segment_start_seconds: float,
    segment_end_seconds: float,
    view_id: str,
) -> list[int]:
    if frame_count <= 0 or duration_seconds <= 0:
        raise ValueError("video frame count and duration must be positive")
    start = max(0.0, float(segment_start_seconds))
    end = min(float(duration_seconds), float(segment_end_seconds))
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"invalid full-video bound [{start}, {end}]")
    first = max(0, math.ceil(start * frame_count / duration_seconds - 0.5))
    last = min(frame_count - 1, math.floor(end * frame_count / duration_seconds - 0.5))
    if last < first:
        first = last = min(frame_count - 1, max(0, frame_count // 2))
    size = last - first + 1

    def uniform(count: int, phase: bool = False) -> list[int]:
        if phase:
            positions = [min(size - 1, int((index + 0.5) * size / count)) for index in range(count)]
        elif count == 1:
            positions = [size // 2]
        else:
            positions = [round(index * (size - 1) / (count - 1)) for index in range(count)]
        return [first + position for position in positions]

    if view_id == "base":
        return uniform(16)
    if view_id == "invariant_nuisance":
        return uniform(8, phase=True)
    if view_id == "dense_acquisition":
        return uniform(24)
    if view_id == "temporal_order_destruction":
        return list(reversed(uniform(8)))
    raise ValueError(f"unknown HERBench view {view_id!r}")


base.sample_frame_indices = sample_frame_indices
_base_record = base.base_record


def base_record(*args, **kwargs):
    record = _base_record(*args, **kwargs)
    question = args[0].question
    record["schema_version"] = base.SCHEMA_VERSION
    record["dataset"] = "HERBench-Lite-v2"
    record["dataset_repository"] = question["dataset_repository"]
    record["dataset_revision"] = question["dataset_revision"]
    record["split"] = question["split"]
    record["video_id"] = question["video_id"]
    record["question_type"] = question["task"]
    record["source_dataset"] = question["source_dataset"]
    record["gold_candidate_index"] = None
    return record


base.base_record = base_record


if __name__ == "__main__":
    raise SystemExit(base.main())
