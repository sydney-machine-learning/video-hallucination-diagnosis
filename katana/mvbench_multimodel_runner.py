#!/usr/bin/env python3
"""Fresh fixed-revision Qwen/InternVL/LLaVA scoring on the MVBench panel."""

from __future__ import annotations

import argparse
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


SICR_KATANA_CANDIDATES = (
    Path.home() / "sicr_route" / "katana",
    Path(__file__).resolve().parents[2] / "sicr_route" / "katana",
)
SICR_KATANA = next(
    (path for path in SICR_KATANA_CANDIDATES if path.is_dir()),
    SICR_KATANA_CANDIDATES[0],
)
if str(SICR_KATANA) not in sys.path:
    sys.path.insert(0, str(SICR_KATANA))

from nextgqa_qwen_runner import (  # noqa: E402
    ShardedJsonlStore,
    decode_selected_frames as decode_selected_frames_sequential,
    probe_video,
    sha256_file,
    write_checkpoint,
)
from star_multimodel_runner import (  # noqa: E402
    BACKENDS,
    load_adapter,
)


SCHEMA_VERSION = "caver-mvbench-fresh-inference-1.0"
SEED = 20260731
VIEW_SPECS = {
    "base": ("reference", "uniform_official_bound_8_frames", 8),
    "invariant_nuisance": (
        "invariant",
        "phase_shift_official_bound_8_frames",
        8,
    ),
    "dense_acquisition": (
        "acquisition",
        "uniform_official_bound_16_frames",
        16,
    ),
    "temporal_order_destruction": (
        "evidence_removal",
        "deterministic_reverse_of_base_frames",
        8,
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample_frame_indices(
    frame_count: int,
    duration_seconds: float,
    segment_start_seconds: float,
    segment_end_seconds: float,
    view_id: str,
) -> list[int]:
    """Match the original view sampler without materialising every frame index."""

    if frame_count <= 0 or duration_seconds <= 0:
        raise ValueError("video frame count and duration must be positive")
    start = max(0.0, float(segment_start_seconds))
    end = min(duration_seconds, float(segment_end_seconds))
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"invalid official bound [{start}, {end}]")
    first = max(0, math.ceil(start * frame_count / duration_seconds - 0.5))
    last = min(
        frame_count - 1,
        math.floor(end * frame_count / duration_seconds - 0.5),
    )
    if last < first:
        centre = min(
            frame_count - 1,
            max(0, int(((start + end) / 2) * frame_count / duration_seconds)),
        )
        first = last = centre
    size = last - first + 1

    def evenly_sample(count: int, phase: bool = False) -> list[int]:
        if phase:
            positions = [
                min(size - 1, int((index + 0.5) * size / count))
                for index in range(count)
            ]
        elif count == 1:
            positions = [size // 2]
        else:
            positions = [
                round(index * (size - 1) / (count - 1))
                for index in range(count)
            ]
        return [first + position for position in positions]

    if view_id == "base":
        return evenly_sample(8)
    if view_id == "invariant_nuisance":
        return evenly_sample(8, phase=True)
    if view_id == "dense_acquisition":
        return evenly_sample(16)
    if view_id == "temporal_order_destruction":
        return list(reversed(evenly_sample(8)))
    raise ValueError(f"unknown view: {view_id!r}")


def decode_selected_frames_seek(
    path: Path,
    indices: Sequence[int],
    frame_count: int,
    duration_seconds: float,
) -> dict[int, Any]:
    """Decode frames by timestamp seeks; fall back to the audited sequential path."""

    import av

    wanted = sorted(set(int(index) for index in indices))
    if not wanted:
        raise ValueError("No requested frame indices")
    try:
        decoded: dict[int, Any] = {}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise ValueError("video stream has no time base")
            time_base = float(stream.time_base)
            if not math.isfinite(time_base) or time_base <= 0:
                raise ValueError("invalid video time base")
            fps = (
                float(stream.average_rate)
                if stream.average_rate is not None
                else frame_count / duration_seconds
            )
            max_seek_decode = max(300, math.ceil(30.0 * fps))
            for requested_index in wanted:
                target_seconds = (
                    (requested_index + 0.5) * duration_seconds / frame_count
                )
                target_pts = max(0, int(target_seconds / time_base))
                container.seek(
                    target_pts,
                    stream=stream,
                    any_frame=False,
                    backward=True,
                )
                best_frame = None
                best_delta = math.inf
                for decoded_count, frame in enumerate(container.decode(stream)):
                    if decoded_count >= max_seek_decode:
                        break
                    if frame.pts is None:
                        continue
                    frame_seconds = float(frame.pts * stream.time_base)
                    delta = abs(frame_seconds - target_seconds)
                    if delta < best_delta:
                        best_frame = frame
                        best_delta = delta
                    if frame_seconds >= target_seconds:
                        break
                if best_frame is None:
                    raise ValueError(
                        f"seek produced no frame for index {requested_index}"
                    )
                decoded[requested_index] = best_frame.to_ndarray(format="rgb24")
        return decoded
    except Exception:
        return decode_selected_frames_sequential(path, wanted)


def build_prompt(question: str, candidates: Sequence[str]) -> tuple[str, list[str]]:
    if not 2 <= len(candidates) <= 10:
        raise ValueError("MVBench candidate count must lie in [2, 10]")
    labels = [chr(ord("A") + index) for index in range(len(candidates))]
    options = "\n".join(
        f"{label}. {text}"
        for label, text in zip(labels, candidates, strict=True)
    )
    prompt = (
        f"Question: {question.strip()}\nOptions:\n{options}\n"
        f"Answer with exactly one option label ({', '.join(labels)}) "
        "and no other text."
    )
    return prompt, labels


@dataclass(frozen=True)
class WorkItem:
    ordinal: int
    question: dict[str, Any]
    view_id: str

    @property
    def key(self) -> str:
        return f"{self.question['question_id']}::{self.view_id}"


def load_manifest(
    path: Path,
    roles: set[str],
    views: Sequence[str],
    max_questions: int | None,
) -> list[WorkItem]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "question_id",
                "task",
                "role",
                "video_path",
                "question",
                "candidates",
                "start",
                "end",
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"line {line_number}: missing {missing}")
            if row["role"] not in {"policy", "calibration", "test"}:
                raise ValueError(f"line {line_number}: invalid role")
            if row["role"] not in roles:
                continue
            if not isinstance(row["candidates"], list):
                raise ValueError(f"line {line_number}: candidates is not a list")
            build_prompt(str(row["question"]), list(map(str, row["candidates"])))
            rows.append(row)
    if max_questions is not None:
        if max_questions <= 0:
            raise ValueError("--max-questions must be positive")
        rows = rows[:max_questions]
    if not rows:
        raise ValueError("manifest selection contains no questions")
    identifiers = [str(row["question_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate question_id")
    items: list[WorkItem] = []
    for row in rows:
        for view_id in views:
            items.append(WorkItem(len(items), row, view_id))
    return items


def base_record(
    item: WorkItem,
    backend: str,
    model_id: str,
    revision: str,
    prompt: str,
    labels: Sequence[str],
    video_path: Path,
    media_sha256: str | None,
    indices: Sequence[int] | None,
    timestamps: Sequence[float] | None,
) -> dict[str, Any]:
    contract, generator, budget = VIEW_SPECS[item.view_id]
    question = item.question
    return {
        "schema_version": SCHEMA_VERSION,
        "status": None,
        "created_at_utc": utc_now(),
        "record_key": item.key,
        "ordinal": item.ordinal,
        "dataset": "MVBench",
        "dataset_revision": (
            "OpenGVLab/MVBench@230a2d4fac8900333c61754641c7a13e069ac9c6"
        ),
        "role": question["role"],
        "question_id": question["question_id"],
        "pilot_id": question["question_id"],
        "question_type": question["task"],
        "model_backend": backend,
        "model_id": model_id,
        "model_revision": revision,
        "processor_revision": revision,
        "prompt": prompt,
        "prompt_template_sha256": sha256_text(prompt),
        "chat_template_sha256": None,
        "media_path": str(video_path),
        "media_sha256": media_sha256,
        "view_id": item.view_id,
        "intervention_contract": contract,
        "view_generator": generator,
        "frame_budget": budget,
        "frame_indices": list(indices) if indices is not None else None,
        "frame_timestamps_seconds": (
            list(timestamps) if timestamps is not None else None
        ),
        "segment_start_seconds": question["start"],
        "segment_end_seconds": question["end"],
        "random_seed": SEED,
        "decoding": {
            "do_sample": False,
            "temperature": None,
            "generated": False,
        },
        "question": question["question"],
        "ordered_candidate_labels": list(labels),
        "ordered_candidate_texts": list(question["candidates"]),
        "gold_candidate_index": None,
        "candidate_sequence_scores": None,
        "candidate_probabilities": None,
        "candidate_token_audit": None,
        "selected_candidate": None,
        "scoring_method": None,
        "parser_status": None,
        "videollm_calls": 0,
        "input_visual_frames": budget,
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


def run(args: argparse.Namespace, adapter: Any | None = None) -> int:
    import numpy as np

    items = load_manifest(
        args.manifest,
        roles=set(args.roles),
        views=args.views,
        max_questions=args.max_questions,
    )
    group_size = len(args.views)
    store = ShardedJsonlStore(
        args.state_dir / "shards",
        args.shard_size,
        args.output_prefix,
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
                        f"missing/duplicate key at {args.output}:{line_number}"
                    )
                store.records[key] = record
    if (store.records or args.output.exists()) and not args.resume:
        raise FileExistsError("output exists; pass --resume")
    expected_keys = {item.key for item in items}
    if args.stop_file is not None and args.stop_file.exists():
        write_checkpoint(store, args.output, args.state_dir, expected_keys)
        return 75

    # Persistent Katana workers may retain a frozen adapter across independent
    # media shards.  The ordinary CLI path still loads exactly one adapter here.
    if adapter is None:
        adapter = load_adapter(args.backend, args.model_id, args.revision)
    np.random.seed(SEED)
    adapter.torch.manual_seed(SEED)
    hashes: dict[Path, str] = {}
    written = 0
    cached_decode_key: tuple[str, float, float, tuple[str, ...]] | None = None
    cached_probe = None
    cached_indices_by_view: dict[str, list[int]] = {}
    cached_decoded: dict[int, Any] = {}

    for offset in range(0, len(items), group_size):
        group = items[offset : offset + group_size]
        if args.stop_file is not None and args.stop_file.exists():
            write_checkpoint(store, args.output, args.state_dir, expected_keys)
            return 75
        pending = [
            item
            for item in group
            if store.records.get(item.key, {}).get("status") != "ok"
        ]
        if not pending:
            continue
        question = group[0].question
        video_path = Path(str(question["video_path"])).resolve()
        prompt, labels = build_prompt(
            str(question["question"]),
            list(map(str, question["candidates"])),
        )
        indices_by_key: dict[str, list[int]] = {}
        decoded: dict[int, Any] = {}
        probe = None
        group_error: BaseException | None = None
        try:
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            fallback_duration = (
                max(float(question["end"]) + 1.0, 1.0)
                if question["end"] is not None
                else 1.0
            )
            probe = probe_video(video_path, fallback_duration, 24.0)
            start = (
                max(0.0, float(question["start"]))
                if question["start"] is not None
                else 0.0
            )
            end = (
                min(probe.duration_seconds, float(question["end"]))
                if question["end"] is not None
                else probe.duration_seconds
            )
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError(f"invalid official bound [{start}, {end}]")
            decode_key = (str(video_path), start, end, tuple(args.views))
            if decode_key != cached_decode_key:
                all_indices_by_view = {
                    view_id: sample_frame_indices(
                        probe.frame_count,
                        probe.duration_seconds,
                        start,
                        end,
                        view_id,
                    )
                    for view_id in args.views
                }
                cached_decoded = decode_selected_frames_seek(
                    video_path,
                    sorted(
                        {
                            index
                            for values in all_indices_by_view.values()
                            for index in values
                        }
                    ),
                    probe.frame_count,
                    probe.duration_seconds,
                )
                cached_decode_key = decode_key
                cached_probe = probe
                cached_indices_by_view = all_indices_by_view
            else:
                probe = cached_probe
            for item in pending:
                indices_by_key[item.key] = cached_indices_by_view[item.view_id]
            decoded = cached_decoded
            if video_path not in hashes:
                hashes[video_path] = sha256_file(video_path)
        except BaseException as exc:
            group_error = exc

        for item in pending:
            indices = indices_by_key.get(item.key)
            timestamps = (
                [
                    (index + 0.5)
                    * probe.duration_seconds
                    / probe.frame_count
                    for index in indices
                ]
                if indices is not None and probe is not None
                else None
            )
            record = base_record(
                item,
                args.backend,
                args.model_id,
                args.revision,
                prompt,
                labels,
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
                    labels,
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
                record["error_type"] = type(exc).__name__
                record["error_message"] = str(exc)
                record["error_traceback"] = "".join(
                    traceback.format_exception(
                        type(exc), exc, exc.__traceback__
                    )
                )[-12000:]
            if record["latency_seconds"] is None:
                record["latency_seconds"] = time.perf_counter() - started
            store.put(record)
            written += 1
            print(
                f"[{item.ordinal + 1}/{len(items)}] {item.key}: "
                f"{record['status']}",
                flush=True,
            )
            if written >= args.checkpoint_every:
                write_checkpoint(
                    store,
                    args.output,
                    args.state_dir,
                    expected_keys,
                )
                written = 0

    write_checkpoint(store, args.output, args.state_dir, expected_keys)
    ok = sum(
        store.records.get(item.key, {}).get("status") == "ok"
        for item in items
    )
    errors = sum(
        store.records.get(item.key, {}).get("status") == "error"
        for item in items
    )
    print(f"Finished: ok={ok}, errors={errors}, total={len(items)}", flush=True)
    return 0 if ok == len(items) and errors == 0 else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--roles",
        default="policy,calibration,test",
        help="comma-separated subset of policy,calibration,test",
    )
    parser.add_argument(
        "--views",
        default=",".join(VIEW_SPECS),
        help="comma-separated ordered view IDs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-prefix")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.roles = tuple(
        value.strip() for value in args.roles.split(",") if value.strip()
    )
    args.views = tuple(
        value.strip() for value in args.views.split(",") if value.strip()
    )
    if not args.roles or not set(args.roles) <= {
        "policy",
        "calibration",
        "test",
    }:
        parser.error("invalid --roles")
    if not args.views or not set(args.views) <= set(VIEW_SPECS):
        parser.error("invalid --views")
    if len(args.views) != len(set(args.views)):
        parser.error("duplicate --views")
    locked = BACKENDS[args.backend]
    args.model_id = args.model_id or locked["model_id"]
    args.revision = args.revision or locked["revision"]
    args.output_prefix = args.output_prefix or f"mvbench-{args.backend}"
    if not 1 <= args.checkpoint_every <= 25:
        parser.error("--checkpoint-every must be between 1 and 25")
    if args.model_id != locked["model_id"] or args.revision != locked["revision"]:
        parser.error("model ID and revision must match the frozen panel")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
