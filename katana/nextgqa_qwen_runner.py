#!/usr/bin/env python3
"""Fresh Qwen3-VL inference for the locked NExT-GQA SICR pilot.

The runner deliberately has no import-time dependency on torch/transformers so
that manifest, sampling, and resume behavior can be tested on a login node.
Model dependencies are imported only when :class:`Qwen3VLAdapter` is created.
"""

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
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATASET_VERSION = "NExT-GQA@63772e0256ad9e34b83d6a108f1ba2041b924607"
SEED = 20260729
LABELS = tuple("ABCDE")
VIEW_GENERATORS = {
    "base": "uniform_full_video_16_frames",
    "invariant_nuisance": "uniform_phase_shift_16_frames",
    "temporal_evidence_sufficiency": "nextgqa_evidence_intervals_16_frames",
    "evidence_removal": "complement_of_nextgqa_evidence_intervals_16_frames",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_softmax(scores: Sequence[float]) -> list[float]:
    if len(scores) < 2 or not all(math.isfinite(value) for value in scores):
        raise ValueError("Candidate scores must contain at least two finite values")
    maximum = max(scores)
    weights = [math.exp(value - maximum) for value in scores]
    total = sum(weights)
    return [value / total for value in weights]


def evenly_sample(indices: Sequence[int], count: int, *, phase: bool = False) -> list[int]:
    """Select ``count`` ordered positions, repeating frames for short videos."""

    if count <= 0:
        raise ValueError("Frame budget must be positive")
    if not indices:
        raise ValueError("Cannot sample an empty frame set")
    size = len(indices)
    if phase:
        positions = [min(size - 1, int((i + 0.5) * size / count)) for i in range(count)]
    elif count == 1:
        positions = [size // 2]
    else:
        positions = [round(i * (size - 1) / (count - 1)) for i in range(count)]
    return [int(indices[position]) for position in positions]


def normalize_intervals(intervals: Iterable[Sequence[float]], duration: float) -> list[list[float]]:
    result: list[list[float]] = []
    for item in intervals:
        if len(item) != 2:
            raise ValueError(f"Malformed evidence interval: {item!r}")
        start, end = float(item[0]), float(item[1])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError(f"Malformed evidence interval: {item!r}")
        start, end = max(0.0, start), min(duration, end)
        if end >= start:
            result.append([start, end])
    return result


def sample_frame_indices(
    frame_count: int,
    duration_seconds: float,
    view_id: str,
    evidence_intervals: Iterable[Sequence[float]],
    budget: int = 16,
) -> list[int]:
    """Return locked frame indices for one of the four SICR views."""

    if frame_count <= 0 or duration_seconds <= 0:
        raise ValueError("Video frame count and duration must be positive")
    all_indices = list(range(frame_count))
    intervals = normalize_intervals(evidence_intervals, duration_seconds)
    if view_id == "base":
        return evenly_sample(all_indices, budget)
    if view_id == "invariant_nuisance":
        return evenly_sample(all_indices, budget, phase=True)

    # Use frame-centre timestamps. Closed intervals preserve boundary evidence.
    times = [(index + 0.5) * duration_seconds / frame_count for index in all_indices]
    evidence = [
        index
        for index, timestamp in zip(all_indices, times, strict=True)
        if any(start <= timestamp <= end for start, end in intervals)
    ]
    if view_id == "temporal_evidence_sufficiency":
        if not evidence:
            raise ValueError("Evidence-only view has no frames inside its intervals")
        return evenly_sample(evidence, budget)
    if view_id == "evidence_removal":
        evidence_set = set(evidence)
        complement = [index for index in all_indices if index not in evidence_set]
        if not complement:
            raise ValueError("Evidence-removal view has an empty complement")
        return evenly_sample(complement, budget)
    raise ValueError(f"Unknown view_id: {view_id!r}")


def safe_video_path(video_root: Path, relative_stem: str) -> Path:
    stem = relative_stem.strip().replace("\\", "/")
    pure = PurePosixPath(stem)
    if not stem or pure.is_absolute() or ".." in pure.parts or pure.suffix:
        raise ValueError(f"Unsafe video_relative_stem: {relative_stem!r}")
    root = video_root.resolve()
    path = root.joinpath(*pure.parts).with_suffix(".mp4").resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Video path escapes root: {relative_stem!r}")
    return path


def load_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


@dataclass(frozen=True)
class WorkItem:
    ordinal: int
    question: dict[str, str]
    view: dict[str, str]

    @property
    def key(self) -> str:
        return f"{self.question['pilot_id']}::{self.view['view_id']}"


def load_work_items(question_csv: Path, view_csv: Path, limit: int | None) -> list[WorkItem]:
    questions = load_csv(
        question_csv,
        {
            "pilot_id", "dataset", "split", "fold", "video_id",
            "video_relative_stem", "qid", "question_type", "duration_seconds",
            "fps", "question", "candidate_ids_json", "candidate_texts_json",
            "gold_candidate_index", "evidence_intervals_json",
        },
    )
    views = load_csv(
        view_csv,
        {
            "pilot_id", "view_id", "contract", "generator", "frame_budget",
            "evidence_intervals_json",
        },
    )
    question_by_id = {row["pilot_id"]: row for row in questions}
    if len(question_by_id) != len(questions):
        raise ValueError("Duplicate pilot_id in question manifest")
    selected_ids = set(list(question_by_id)[:limit] if limit is not None else question_by_id)
    views_by_id: dict[str, dict[str, dict[str, str]]] = {}
    for row in views:
        pilot_id, view_id = row["pilot_id"], row["view_id"]
        if pilot_id not in question_by_id:
            raise ValueError(f"View references unknown pilot_id: {pilot_id}")
        if view_id not in VIEW_GENERATORS:
            raise ValueError(f"Unexpected view_id: {view_id}")
        if row["generator"] != VIEW_GENERATORS[view_id]:
            raise ValueError(f"Generator mismatch for {pilot_id}/{view_id}")
        views_by_id.setdefault(pilot_id, {})
        if view_id in views_by_id[pilot_id]:
            raise ValueError(f"Duplicate view: {pilot_id}/{view_id}")
        views_by_id[pilot_id][view_id] = row

    items: list[WorkItem] = []
    ordinal = 0
    for question in questions:
        pilot_id = question["pilot_id"]
        present = views_by_id.get(pilot_id, {})
        if set(present) != set(VIEW_GENERATORS):
            raise ValueError(f"{pilot_id} does not have exactly the four locked views")
        if pilot_id not in selected_ids:
            continue
        for view_id in VIEW_GENERATORS:
            items.append(WorkItem(ordinal, question, present[view_id]))
            ordinal += 1
    return items


class ShardedJsonlStore:
    """Atomic deterministic JSONL shards with replace-in-place resume semantics."""

    def __init__(self, output_dir: Path, shard_size: int, prefix: str = "qwen3vl") -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.prefix = prefix
        self.records: dict[str, dict[str, Any]] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.output_dir.glob(f"{prefix}-part-*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = record.get("record_key")
                    if not key or key in self.records:
                        raise ValueError(f"Missing/duplicate record_key in {path}:{line_number}")
                    self.records[key] = record

    def put(self, record: dict[str, Any]) -> Path:
        key = str(record["record_key"])
        ordinal = int(record["ordinal"])
        self.records[key] = record
        shard_index = ordinal // self.shard_size
        start, end = shard_index * self.shard_size, (shard_index + 1) * self.shard_size
        shard_records = sorted(
            (
                value for value in self.records.values()
                if start <= int(value["ordinal"]) < end
            ),
            key=lambda value: int(value["ordinal"]),
        )
        path = self.output_dir / f"{self.prefix}-part-{shard_index:05d}.jsonl"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for value in shard_records:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return path

    def export(self, path: Path) -> None:
        """Atomically materialize the de-duplicated single-file contract output."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in sorted(
                self.records.values(), key=lambda value: int(value["ordinal"])
            ):
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def write_checkpoint(
    store: ShardedJsonlStore,
    output_path: Path,
    state_dir: Path,
    expected_keys: set[str],
) -> None:
    """Export output and a small progress audit, both via atomic replacement."""

    store.export(output_path)
    relevant = [
        record for key, record in store.records.items() if key in expected_keys
    ]
    audit = {
        "updated_at_utc": utc_now(),
        "output": str(output_path),
        "expected_records": len(expected_keys),
        "present_records": len(relevant),
        "ok_records": sum(record.get("status") == "ok" for record in relevant),
        "error_records": sum(record.get("status") == "error" for record in relevant),
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "progress.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@dataclass
class VideoProbe:
    frame_count: int
    duration_seconds: float
    fps: float


def probe_video(path: Path, fallback_duration: float, fallback_fps: float) -> VideoProbe:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else fallback_fps
        frame_count = int(stream.frames or 0)
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else fallback_duration
        )
    if frame_count <= 0:
        frame_count = max(1, round(duration * fps))
    return VideoProbe(frame_count, duration, fps)


def decode_selected_frames(path: Path, indices: Sequence[int]) -> dict[int, Any]:
    """Decode only retained RGB frames while traversing the clip once."""

    import av

    wanted = set(int(index) for index in indices)
    if not wanted:
        raise ValueError("No requested frame indices")
    decoded: dict[int, Any] = {}
    last_frame = None
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            last_frame = frame
            if index in wanted:
                decoded[index] = frame.to_ndarray(format="rgb24")
            if index >= max(wanted):
                break
    if last_frame is None:
        raise ValueError(f"Video contains no decodable frames: {path}")
    missing = sorted(wanted.difference(decoded))
    if missing:
        # Stream metadata occasionally overstates frame count. Map only trailing
        # requests to the last real frame, retaining requested indices in audit.
        last_array = last_frame.to_ndarray(format="rgb24")
        for index in missing:
            decoded[index] = last_array
    return decoded


def build_prompt(question: str, candidate_texts: Sequence[str]) -> str:
    if len(candidate_texts) < 2 or len(candidate_texts) > len(LABELS):
        raise ValueError("Expected between two and five answer candidates")
    options = "\n".join(
        f"{LABELS[index]}. {text}" for index, text in enumerate(candidate_texts)
    )
    return (
        f"Question: {question.strip()}\n"
        f"Options:\n{options}\n"
        "Answer with exactly one option label (A, B, C, D, or E) and no other text."
    )


class Qwen3VLAdapter:
    def __init__(self, model_id: str, revision: str) -> None:
        import torch
        from huggingface_hub import model_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if model_id == MODEL_ID and revision != MODEL_REVISION:
            raise ValueError(
                f"{MODEL_ID} is frozen at revision {MODEL_REVISION}; "
                f"refusing {revision!r}"
            )
        self.torch = torch
        self.model_id = model_id
        self.resolved_revision = model_info(model_id, revision=revision).sha
        if not self.resolved_revision:
            raise RuntimeError("Could not resolve an immutable model revision")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=self.resolved_revision
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=self.resolved_revision,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map="auto",
        ).eval()
        self.device = next(self.model.parameters()).device
        self.tokenizer = self.processor.tokenizer
        template = str(
            getattr(self.processor, "chat_template", None)
            or getattr(self.tokenizer, "chat_template", "")
        )
        self.chat_template_sha256 = sha256_bytes(template.encode("utf-8"))

    def _to_device(self, inputs: Any) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, self.torch.Tensor) else value
            for key, value in dict(inputs).items()
        }

    def _candidate_tokens(self, labels: Sequence[str]) -> list[list[int]]:
        tokenized: list[list[int]] = []
        for label in labels:
            ids = list(self.tokenizer(label, add_special_tokens=False).input_ids)
            if not ids:
                raise RuntimeError(f"Candidate label {label!r} has no tokens")
            tokenized.append(ids)
        return tokenized

    def _prefix(self, frames: Sequence[Any], prompt: str, fps: float) -> dict[str, Any]:
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": list(frames), "fps": float(fps)},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return self._to_device(inputs)

    def _fallback_scores(
        self, prefix: dict[str, Any], labels: Sequence[str], token_ids: list[list[int]]
    ) -> tuple[list[float], list[dict[str, Any]]]:
        torch = self.torch
        prefix_length = int(prefix["input_ids"].shape[1])
        scores, audits = [], []
        for label, continuation in zip(labels, token_ids, strict=True):
            extra = torch.tensor(
                [continuation],
                dtype=prefix["input_ids"].dtype,
                device=prefix["input_ids"].device,
            )
            inputs = dict(prefix)
            inputs["input_ids"] = torch.cat([prefix["input_ids"], extra], dim=1)
            attention = torch.ones(
                (1, len(continuation)),
                dtype=prefix["attention_mask"].dtype,
                device=prefix["attention_mask"].device,
            )
            inputs["attention_mask"] = torch.cat(
                [prefix["attention_mask"], attention], dim=1
            )
            output = self.model(**inputs, use_cache=False, return_dict=True)
            log_probs = torch.log_softmax(output.logits.float(), dim=-1)
            per_token = [
                float(log_probs[0, prefix_length - 1 + offset, token_id].cpu())
                for offset, token_id in enumerate(continuation)
            ]
            score = float(sum(per_token))
            scores.append(score)
            audits.append({
                "candidate": label,
                "token_ids": continuation,
                "tokens": self.tokenizer.convert_ids_to_tokens(continuation),
                "per_token_log_probabilities": per_token,
                "sequence_log_probability_sum": score,
                "mean_token_log_probability": score / len(continuation),
                "token_count": len(continuation),
            })
            del output, log_probs
        return scores, audits

    def score(
        self, frames: Sequence[Any], prompt: str, labels: Sequence[str], fps: float
    ) -> dict[str, Any]:
        torch = self.torch
        prefix = self._prefix(frames, prompt, fps)
        prefix_length = int(prefix["input_ids"].shape[1])
        token_ids = self._candidate_tokens(labels)
        started = time.perf_counter()
        if all(len(ids) == 1 for ids in token_ids):
            output = self.model(**prefix, use_cache=False, return_dict=True)
            log_probs = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
            scores = [float(log_probs[ids[0]].cpu()) for ids in token_ids]
            audits = [
                {
                    "candidate": label,
                    "token_ids": ids,
                    "tokens": self.tokenizer.convert_ids_to_tokens(ids),
                    "per_token_log_probabilities": [score],
                    "sequence_log_probability_sum": score,
                    "mean_token_log_probability": score,
                    "token_count": 1,
                }
                for label, ids, score in zip(labels, token_ids, scores, strict=True)
            ]
            method, calls = "single_prefix_next_token_logits", 1
            del output, log_probs
        else:
            scores, audits = self._fallback_scores(prefix, labels, token_ids)
            method, calls = "teacher_forced_sequence_fallback", len(labels)
        probabilities = stable_softmax(scores)
        grids = prefix.get("video_grid_thw")
        grid_values = grids.detach().cpu().tolist() if grids is not None else None
        return {
            "candidate_sequence_scores": scores,
            "candidate_probabilities": probabilities,
            "candidate_token_audit": audits,
            "selected_candidate": labels[max(range(len(scores)), key=scores.__getitem__)],
            "scoring_method": method,
            "videollm_calls": calls,
            "input_language_tokens": prefix_length,
            "output_language_tokens": 0,
            "video_grid_thw": grid_values,
            "visual_tokens": None,
            "latency_seconds": time.perf_counter() - started,
        }


def base_record(
    item: WorkItem,
    model_id: str,
    revision: str | None,
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
        "schema_version": "sicr-fresh-inference-1.0",
        "status": None,
        "created_at_utc": utc_now(),
        "record_key": item.key,
        "ordinal": item.ordinal,
        "dataset": question["dataset"],
        "dataset_version": DATASET_VERSION,
        "split": question["split"],
        "fold": int(question["fold"]),
        "pilot_id": question["pilot_id"],
        "source_clip_id": question["video_id"],
        "question_id": question["qid"],
        "question_type": question["question_type"],
        "model_provider": "Hugging Face",
        "model_id": model_id,
        "model_revision": revision,
        "processor_revision": revision,
        "prompt": prompt,
        "prompt_template_sha256": sha256_bytes(prompt.encode("utf-8")),
        "chat_template_sha256": None,
        "media_path": str(video_path),
        "media_sha256": media_sha256,
        "view_id": view["view_id"],
        "intervention_contract": view["contract"],
        "view_generator": view["generator"],
        "frame_budget": int(view["frame_budget"]),
        "frame_indices": list(indices) if indices is not None else None,
        "frame_timestamps_seconds": list(timestamps) if timestamps is not None else None,
        "evidence_intervals_seconds": json.loads(view["evidence_intervals_json"]),
        "random_seed": SEED,
        "decoding": {"do_sample": False, "temperature": None, "generated": False},
        "question": question["question"],
        "ordered_candidate_ids": candidate_ids,
        "ordered_candidate_labels": list(LABELS[:len(candidate_texts)]),
        "ordered_candidate_texts": candidate_texts,
        "gold_candidate_index_audit_only": int(question["gold_candidate_index"]),
        "candidate_sequence_scores": None,
        "candidate_probabilities": None,
        "candidate_token_audit": None,
        "selected_candidate": None,
        "scoring_method": None,
        "parser_status": None,
        "invalid_response_reason": None,
        "self_reported_confidence": None,
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
    shard_dir = args.state_dir / "shards"
    store = ShardedJsonlStore(shard_dir, args.shard_size, args.output_prefix)
    if args.output.is_file() and not store.records:
        # Permit recovery if the state directory was copied without its shards.
        with args.output.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record.get("record_key")
                if not key or key in store.records:
                    raise ValueError(
                        f"Missing/duplicate record_key in {args.output}:{line_number}"
                    )
                store.records[key] = record
    if (store.records or args.output.exists()) and not args.resume:
        raise FileExistsError(
            f"{args.output} already contains output; pass --resume to continue"
        )
    expected_keys = {item.key for item in items}
    if args.stop_file is not None and args.stop_file.exists():
        write_checkpoint(store, args.output, args.state_dir, expected_keys)
        print(f"Stop file present before model load: {args.stop_file}", flush=True)
        return 75
    adapter = Qwen3VLAdapter(args.model_id, args.revision)
    np.random.seed(SEED)
    adapter.torch.manual_seed(SEED)
    media_hashes: dict[Path, str] = {}
    completed = sum(
        store.records.get(item.key, {}).get("status") == "ok" for item in items
    )
    print(f"Loaded {len(items)} work items; {completed} already complete", flush=True)
    written_since_checkpoint = 0

    # Items are manifest ordered: process four views after a single video decode.
    for offset in range(0, len(items), len(VIEW_GENERATORS)):
        group = items[offset:offset + len(VIEW_GENERATORS)]
        if args.stop_file is not None and args.stop_file.exists():
            write_checkpoint(store, args.output, args.state_dir, expected_keys)
            print(f"Stop file detected: {args.stop_file}", flush=True)
            return 75
        question = group[0].question
        pending = [
            item for item in group
            if store.records.get(item.key, {}).get("status") != "ok"
        ]
        if not pending:
            continue
        video_path = safe_video_path(args.video_root, question["video_relative_stem"])
        candidate_texts = json.loads(question["candidate_texts_json"])
        prompt = build_prompt(question["question"], candidate_texts)
        indices_by_key: dict[str, list[int]] = {}
        decoded: dict[int, Any] = {}
        probe = None
        group_error: BaseException | None = None
        try:
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            probe = probe_video(
                video_path, float(question["duration_seconds"]), float(question["fps"])
            )
            for item in pending:
                intervals = json.loads(item.view["evidence_intervals_json"])
                indices_by_key[item.key] = sample_frame_indices(
                    probe.frame_count,
                    probe.duration_seconds,
                    item.view["view_id"],
                    intervals,
                    int(item.view["frame_budget"]),
                )
            decoded = decode_selected_frames(
                video_path,
                sorted({index for values in indices_by_key.values() for index in values}),
            )
            media_hashes.setdefault(video_path, sha256_file(video_path))
        except BaseException as exc:  # Persist one complete error row per view.
            group_error = exc

        for item in pending:
            indices = indices_by_key.get(item.key)
            timestamps = (
                [(index + 0.5) * probe.duration_seconds / probe.frame_count for index in indices]
                if indices is not None and probe is not None else None
            )
            record = base_record(
                item,
                args.model_id,
                adapter.resolved_revision,
                prompt,
                video_path,
                media_hashes.get(video_path),
                indices,
                timestamps,
            )
            record["chat_template_sha256"] = adapter.chat_template_sha256
            try:
                if group_error is not None:
                    raise group_error
                assert indices is not None and probe is not None
                frames = [decoded[index] for index in indices]
                result = adapter.score(
                    frames,
                    prompt,
                    LABELS[:len(candidate_texts)],
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
            store.put(record)
            written_since_checkpoint += 1
            print(
                f"[{item.ordinal + 1}/{len(items)}] {item.key}: {record['status']}",
                flush=True,
            )
            if written_since_checkpoint >= args.checkpoint_every:
                write_checkpoint(store, args.output, args.state_dir, expected_keys)
                written_since_checkpoint = 0
            if args.stop_file is not None and args.stop_file.exists():
                write_checkpoint(store, args.output, args.state_dir, expected_keys)
                print(f"Stop file detected: {args.stop_file}", flush=True)
                return 75
    write_checkpoint(store, args.output, args.state_dir, expected_keys)
    ok = sum(store.records.get(item.key, {}).get("status") == "ok" for item in items)
    errors = sum(store.records.get(item.key, {}).get("status") == "error" for item in items)
    print(f"Finished: ok={ok}, errors={errors}, total={len(items)}", flush=True)
    return 0 if errors == 0 and ok == len(items) else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument(
        "--manifest", "--views", dest="views", type=Path, required=True
    )
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="qwen3vl")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument(
        "--max-questions", "--limit", dest="limit", type=int,
        help="Run the first N questions (use 20 as gate); omit for the full manifest",
    )
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--max-questions must be positive")
    if not 10 <= args.checkpoint_every <= 25:
        parser.error("--checkpoint-every must be between 10 and 25")
    if args.model_id == MODEL_ID and args.revision != MODEL_REVISION:
        parser.error(
            f"{MODEL_ID} is frozen at revision {MODEL_REVISION}; "
            f"refusing {args.revision!r}"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
