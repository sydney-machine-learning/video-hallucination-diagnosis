#!/usr/bin/env python3
"""Fresh Qwen, InternVL, or LLaVA inference on the locked STAR pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


KATANA_DIR = Path(__file__).resolve().parent
if str(KATANA_DIR) not in sys.path:
    sys.path.insert(0, str(KATANA_DIR))

from nextgqa_qwen_runner import (  # noqa: E402
    Qwen3VLAdapter,
    ShardedJsonlStore,
    decode_selected_frames,
    probe_video,
    safe_video_path,
    sha256_file,
    write_checkpoint,
)


SCHEMA_VERSION = "sicr-fresh-inference-1.0"
SEED = 20260731
LABELS = tuple("ABCD")
VIEW_SPECS = {
    "base": ("reference", "uniform_official_segment_8_frames", 8),
    "invariant_nuisance": (
        "invariant",
        "phase_shift_official_segment_8_frames",
        8,
    ),
    "dense_acquisition": (
        "acquisition",
        "uniform_official_segment_16_frames",
        16,
    ),
    "temporal_order_destruction": (
        "evidence_removal",
        "deterministic_reverse_of_base_frames",
        8,
    ),
}
BACKENDS = {
    "qwen": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    },
    "internvl": {
        "model_id": "OpenGVLab/InternVL3_5-4B-Instruct",
        "revision": "a3fd3158be4027880d91ccee80ea8089559e0896",
    },
    "llava": {
        "model_id": "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        "revision": "802f16c8346a062fef7ddf92b2bd64a770a50a1a",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evenly_sample(indices: Sequence[int], count: int, phase: bool = False) -> list[int]:
    if count <= 0 or not indices:
        raise ValueError("sampling requires non-empty indices and positive count")
    size = len(indices)
    if phase:
        positions = [min(size - 1, int((i + 0.5) * size / count)) for i in range(count)]
    elif count == 1:
        positions = [size // 2]
    else:
        positions = [round(i * (size - 1) / (count - 1)) for i in range(count)]
    return [int(indices[position]) for position in positions]


def sample_star_frame_indices(
    frame_count: int,
    duration_seconds: float,
    segment_start_seconds: float,
    segment_end_seconds: float,
    view_id: str,
) -> list[int]:
    if frame_count <= 0 or duration_seconds <= 0:
        raise ValueError("video frame count and duration must be positive")
    start = max(0.0, float(segment_start_seconds))
    end = min(duration_seconds, float(segment_end_seconds))
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"invalid STAR official segment [{start}, {end}]")
    indices = [
        index
        for index in range(frame_count)
        if start
        <= (index + 0.5) * duration_seconds / frame_count
        <= end
    ]
    if not indices:
        centre = min(
            frame_count - 1,
            max(0, int(((start + end) / 2) * frame_count / duration_seconds)),
        )
        indices = [centre]
    if view_id == "base":
        return evenly_sample(indices, 8)
    if view_id == "invariant_nuisance":
        return evenly_sample(indices, 8, phase=True)
    if view_id == "dense_acquisition":
        return evenly_sample(indices, 16)
    if view_id == "temporal_order_destruction":
        return list(reversed(evenly_sample(indices, 8)))
    raise ValueError(f"unknown STAR view: {view_id!r}")


def build_prompt(question: str, candidates: Sequence[str]) -> str:
    if len(candidates) != 4:
        raise ValueError("STAR requires exactly four candidates")
    options = "\n".join(
        f"{label}. {text}" for label, text in zip(LABELS, candidates, strict=True)
    )
    return (
        f"Question: {question.strip()}\nOptions:\n{options}\n"
        "Answer with exactly one option label (A, B, C, or D) and no other text."
    )


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


@dataclass(frozen=True)
class WorkItem:
    ordinal: int
    question: dict[str, str]
    view: dict[str, str]

    @property
    def key(self) -> str:
        return f"{self.question['pilot_id']}::{self.view['view_id']}"


def load_work_items(
    question_csv: Path, view_csv: Path, limit: int | None
) -> list[WorkItem]:
    questions = read_csv(
        question_csv,
        {
            "pilot_id",
            "dataset",
            "dataset_version",
            "split",
            "fold",
            "video_id",
            "video_relative_stem",
            "question_id",
            "question_type",
            "segment_start_seconds",
            "segment_end_seconds",
            "question",
            "candidate_ids_json",
            "candidate_labels_json",
            "candidate_texts_json",
            "gold_candidate_index",
            "evidence_intervals_json",
            "evidence_interval_status",
        },
    )
    views = read_csv(
        view_csv,
        {
            "pilot_id",
            "view_id",
            "contract",
            "generator",
            "frame_budget",
            "segment_start_seconds",
            "segment_end_seconds",
            "evidence_intervals_json",
            "evidence_interval_status",
            "mechanism",
        },
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("--max-questions must be positive")
        questions = questions[:limit]
    question_by_id: dict[str, dict[str, str]] = {}
    for question in questions:
        pilot_id = question["pilot_id"]
        if not pilot_id or pilot_id in question_by_id:
            raise ValueError(f"duplicate/empty pilot_id: {pilot_id!r}")
        labels = json.loads(question["candidate_labels_json"])
        texts = json.loads(question["candidate_texts_json"])
        if labels != list(LABELS) or len(texts) != 4:
            raise ValueError(f"{pilot_id}: STAR candidate contract mismatch")
        if json.loads(question["evidence_intervals_json"]) != []:
            raise ValueError(f"{pilot_id}: answer-evidence intervals must remain empty")
        if question["evidence_interval_status"] != "unavailable_not_fabricated":
            raise ValueError(f"{pilot_id}: evidence status mismatch")
        question_by_id[pilot_id] = question

    views_by_id: dict[str, dict[str, dict[str, str]]] = {
        pilot_id: {} for pilot_id in question_by_id
    }
    for view in views:
        pilot_id = view["pilot_id"]
        if pilot_id not in question_by_id:
            continue
        view_id = view["view_id"]
        if view_id not in VIEW_SPECS or view_id in views_by_id[pilot_id]:
            raise ValueError(f"{pilot_id}: unsupported/duplicate view {view_id!r}")
        contract, generator, budget = VIEW_SPECS[view_id]
        if (
            view["contract"] != contract
            or view["generator"] != generator
            or int(view["frame_budget"]) != budget
        ):
            raise ValueError(f"{pilot_id}/{view_id}: view contract mismatch")
        if json.loads(view["evidence_intervals_json"]) != []:
            raise ValueError(f"{pilot_id}/{view_id}: fabricated evidence interval")
        views_by_id[pilot_id][view_id] = view

    items: list[WorkItem] = []
    ordinal = 0
    for question in questions:
        pilot_id = question["pilot_id"]
        present = views_by_id[pilot_id]
        if set(present) != set(VIEW_SPECS):
            raise ValueError(f"{pilot_id}: expected exactly four locked views")
        for view_id in VIEW_SPECS:
            items.append(WorkItem(ordinal, question, present[view_id]))
            ordinal += 1
    return items


class InternVLAdapter:
    def __init__(self, model_id: str, revision: str) -> None:
        import nextgqa_internvl_runner as internvl

        self.impl = internvl
        self.model_id = model_id
        self.resolved_revision = revision
        self.model, self.tokenizer, self.dtype, self.device = internvl.load_model(
            model_id, revision
        )
        self.torch = __import__("torch")
        template = str(getattr(self.model, "conv_template", ""))
        self.chat_template_sha256 = sha256_text(template)

    def score(
        self, frames: Sequence[Any], prompt: str, labels: Sequence[str], fps: float
    ) -> dict[str, Any]:
        del fps
        from PIL import Image

        images = [
            frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
            for frame in frames
        ]
        pixels = self.impl.preprocess_frames(
            images, self.dtype, self.device
        )
        query, input_ids, packed = self.impl.build_query(
            self.model, self.tokenizer, prompt, [1] * len(images)
        )
        attention_mask, image_flags = packed
        previous = self.impl.OPTION_LABELS
        self.impl.OPTION_LABELS = tuple(labels)
        try:
            scores = self.impl.score_labels(
                self.model,
                self.tokenizer,
                pixels,
                input_ids,
                attention_mask,
                image_flags,
                query,
            )
        finally:
            self.impl.OPTION_LABELS = previous
        probabilities = scores["candidate_probabilities"]
        selected = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "candidate_sequence_scores": scores[
                "candidate_sequence_log_probabilities"
            ],
            "candidate_probabilities": probabilities,
            "candidate_token_audit": [
                {
                    "candidate": label,
                    "token_ids": ids,
                    "tokens": tokens,
                    "per_token_log_probabilities": token_scores,
                    "sequence_log_probability_sum": sum(token_scores),
                    "mean_token_log_probability": sum(token_scores)
                    / len(token_scores),
                    "token_count": len(token_scores),
                }
                for label, ids, tokens, token_scores in zip(
                    labels,
                    scores["candidate_token_ids"],
                    scores["candidate_token_strings"],
                    scores["candidate_token_log_probabilities"],
                    strict=True,
                )
            ],
            "selected_candidate": labels[selected],
            "scoring_method": (
                "single_prefix_next_token_logits"
                if scores["candidate_scoring_forward_passes"] == 1
                else "teacher_forced_sequence_fallback"
            ),
            "videollm_calls": scores["candidate_scoring_forward_passes"],
            "input_language_tokens": int(input_ids.shape[1]),
            "output_language_tokens": 0,
            "visual_tokens": None,
            "latency_seconds": None,
        }


def load_adapter(backend: str, model_id: str, revision: str) -> Any:
    expected = BACKENDS[backend]
    if model_id != expected["model_id"] or revision != expected["revision"]:
        raise ValueError(f"{backend} model/revision differs from frozen panel")
    if backend == "qwen":
        return Qwen3VLAdapter(model_id, revision)
    if backend == "internvl":
        return InternVLAdapter(model_id, revision)
    if backend == "llava":
        from nextgqa_llava_ov2_runner import LlavaOneVision2Adapter

        return LlavaOneVision2Adapter(model_id, revision)
    raise ValueError(f"unknown backend {backend!r}")


def base_record(
    item: WorkItem,
    model_id: str,
    revision: str,
    prompt: str,
    video_path: Path,
    media_sha256: str | None,
    indices: Sequence[int] | None,
    timestamps: Sequence[float] | None,
) -> dict[str, Any]:
    question, view = item.question, item.view
    candidate_ids = json.loads(question["candidate_ids_json"])
    candidate_texts = json.loads(question["candidate_texts_json"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": None,
        "created_at_utc": utc_now(),
        "record_key": item.key,
        "ordinal": item.ordinal,
        "dataset": question["dataset"],
        "dataset_version": question["dataset_version"],
        "split": question["split"],
        "fold": int(question["fold"]),
        "pilot_id": question["pilot_id"],
        "source_clip_id": question["video_id"],
        "video_id": question["video_id"],
        "question_id": question["question_id"],
        "qid": question["question_id"],
        "question_type": question["question_type"],
        "model_provider": "Hugging Face",
        "model_id": model_id,
        "model_revision": revision,
        "processor_revision": revision,
        "prompt": prompt,
        "prompt_template_sha256": sha256_text(prompt),
        "chat_template_sha256": None,
        "media_path": str(video_path),
        "media_sha256": media_sha256,
        "view_id": view["view_id"],
        "intervention_contract": view["contract"],
        "contract": view["contract"],
        "view_generator": view["generator"],
        "generator": view["generator"],
        "mechanism": view["mechanism"],
        "frame_budget": int(view["frame_budget"]),
        "frame_indices": list(indices) if indices is not None else None,
        "frame_timestamps_seconds": list(timestamps) if timestamps is not None else None,
        "segment_start_seconds": float(question["segment_start_seconds"]),
        "segment_end_seconds": float(question["segment_end_seconds"]),
        "evidence_intervals_seconds": [],
        "evidence_interval_status": "unavailable_not_fabricated",
        "random_seed": SEED,
        "decoding": {"do_sample": False, "temperature": None, "generated": False},
        "question": question["question"],
        "ordered_candidate_ids": candidate_ids,
        "ordered_candidate_labels": list(LABELS),
        "ordered_candidate_texts": candidate_texts,
        "gold_candidate_index_audit_only": int(question["gold_candidate_index"]),
        "candidate_sequence_scores": None,
        "candidate_probabilities": None,
        "candidate_token_audit": None,
        "selected_candidate": None,
        "scoring_method": None,
        "parser_status": None,
        "invalid_response_reason": None,
        "videollm_calls": 0,
        "input_visual_frames": int(view["frame_budget"]),
        "visual_tokens": None,
        "input_language_tokens": None,
        "output_language_tokens": 0,
        "latency_seconds": None,
        "peak_accelerator_memory_bytes": None,
        "hostname": platform.node(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "error_type": None,
        "error_message": None,
        "error_traceback": None,
    }


def run(args: argparse.Namespace) -> int:
    import numpy as np

    items = load_work_items(args.questions, args.views, args.limit)
    store = ShardedJsonlStore(
        args.state_dir / "shards", args.shard_size, args.output_prefix
    )
    if args.output.is_file() and not store.records:
        with args.output.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record.get("record_key")
                if not key or key in store.records:
                    raise ValueError(
                        f"missing/duplicate record key at {args.output}:{line_number}"
                    )
                store.records[key] = record
    if (store.records or args.output.exists()) and not args.resume:
        raise FileExistsError("output exists; pass --resume")
    expected_keys = {item.key for item in items}
    if args.stop_file is not None and args.stop_file.exists():
        write_checkpoint(store, args.output, args.state_dir, expected_keys)
        return 75
    adapter = load_adapter(args.backend, args.model_id, args.revision)
    np.random.seed(SEED)
    adapter.torch.manual_seed(SEED)
    hashes: dict[Path, str] = {}
    written = 0

    for offset in range(0, len(items), len(VIEW_SPECS)):
        group = items[offset : offset + len(VIEW_SPECS)]
        if args.stop_file is not None and args.stop_file.exists():
            write_checkpoint(store, args.output, args.state_dir, expected_keys)
            return 75
        pending = [
            item for item in group
            if store.records.get(item.key, {}).get("status") != "ok"
        ]
        if not pending:
            continue
        question = group[0].question
        video_path = safe_video_path(args.video_root, question["video_relative_stem"])
        candidates = json.loads(question["candidate_texts_json"])
        prompt = build_prompt(question["question"], candidates)
        indices_by_key: dict[str, list[int]] = {}
        decoded: dict[int, Any] = {}
        probe = None
        group_error: BaseException | None = None
        try:
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            fallback_duration = max(
                float(question["segment_end_seconds"]) + 1.0, 1.0
            )
            probe = probe_video(video_path, fallback_duration, 24.0)
            for item in pending:
                indices_by_key[item.key] = sample_star_frame_indices(
                    probe.frame_count,
                    probe.duration_seconds,
                    float(item.view["segment_start_seconds"]),
                    float(item.view["segment_end_seconds"]),
                    item.view["view_id"],
                )
            decoded = decode_selected_frames(
                video_path,
                sorted({index for values in indices_by_key.values() for index in values}),
            )
            hashes.setdefault(video_path, sha256_file(video_path))
        except BaseException as exc:
            group_error = exc

        for item in pending:
            indices = indices_by_key.get(item.key)
            timestamps = (
                [
                    (index + 0.5) * probe.duration_seconds / probe.frame_count
                    for index in indices
                ]
                if indices is not None and probe is not None
                else None
            )
            record = base_record(
                item,
                args.model_id,
                args.revision,
                prompt,
                video_path,
                hashes.get(video_path),
                indices,
                timestamps,
            )
            record["chat_template_sha256"] = adapter.chat_template_sha256
            started = time.perf_counter()
            try:
                if group_error is not None:
                    raise group_error
                assert indices is not None and probe is not None
                result = adapter.score(
                    [decoded[index] for index in indices],
                    prompt,
                    LABELS,
                    probe.fps,
                )
                record.update(result)
                record["status"] = "ok"
                record["parser_status"] = "not_applicable_logit_scoring"
                if adapter.torch.cuda.is_available():
                    record["peak_accelerator_memory_bytes"] = (
                        adapter.torch.cuda.max_memory_allocated(adapter.device)
                    )
            except BaseException as exc:
                record["status"] = "error"
                record["parser_status"] = "error"
                record["invalid_response_reason"] = str(exc)
                record["error_type"] = type(exc).__name__
                record["error_message"] = str(exc)
                record["error_traceback"] = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )[-12000:]
            if record["latency_seconds"] is None:
                record["latency_seconds"] = time.perf_counter() - started
            store.put(record)
            written += 1
            print(
                f"[{item.ordinal + 1}/{len(items)}] {item.key}: {record['status']}",
                flush=True,
            )
            if written >= args.checkpoint_every:
                write_checkpoint(store, args.output, args.state_dir, expected_keys)
                written = 0
    write_checkpoint(store, args.output, args.state_dir, expected_keys)
    ok = sum(store.records.get(item.key, {}).get("status") == "ok" for item in items)
    errors = sum(
        store.records.get(item.key, {}).get("status") == "error" for item in items
    )
    print(f"Finished: ok={ok}, errors={errors}, total={len(items)}", flush=True)
    return 0 if ok == len(items) and errors == 0 else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--manifest", "--views", dest="views", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-prefix")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--max-questions", "--limit", dest="limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    locked = BACKENDS[args.backend]
    args.model_id = args.model_id or locked["model_id"]
    args.revision = args.revision or locked["revision"]
    args.output_prefix = args.output_prefix or f"star-{args.backend}"
    if not 1 <= args.checkpoint_every <= 25:
        parser.error("--checkpoint-every must be between 1 and 25")
    if args.model_id != locked["model_id"] or args.revision != locked["revision"]:
        parser.error("model ID and revision must match the frozen backend panel")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
