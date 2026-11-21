#!/usr/bin/env python3
"""Fresh NExT-GQA candidate scoring with InternVL3.5-4B-Instruct.

The runner is deliberately self-contained: it reads the locked question/view
CSVs, decodes exactly the requested video views, and writes one recoverable
JSON shard per (pilot_id, view_id).  The public JSONL is atomically rebuilt at
checkpoints, so a walltime stop cannot corrupt already completed inference.

For the locked A--E answer labels the InternVL tokenizer uses one token per
label.  All five sequence scores are consequently obtained from one
multimodal prefix forward pass.  A correct, slower teacher-forced fallback is
kept for a future tokenizer/revision where a label might span several tokens.
"""

from __future__ import annotations

import argparse
import copy
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


MODEL_ID = "OpenGVLab/InternVL3_5-4B-Instruct"
MODEL_REVISION = "a3fd3158be4027880d91ccee80ea8089559e0896"
SCHEMA_VERSION = "sicr-fresh-inference-1.0"
SEED = 20260729
OPTION_LABELS = ("A", "B", "C", "D", "E")
EXPECTED_GENERATORS = {
    "uniform_full_video_16_frames",
    "uniform_phase_shift_16_frames",
    "nextgqa_evidence_intervals_16_frames",
    "complement_of_nextgqa_evidence_intervals_16_frames",
}


@dataclass(frozen=True)
class WorkItem:
    question: dict[str, str]
    view: dict[str, str]

    @property
    def key(self) -> tuple[str, str]:
        return self.question["pilot_id"], self.view["view_id"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def parse_json_list(raw: str, field: str) -> list[Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {field}: {raw!r}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{field} must contain a JSON list")
    return value


def load_work_items(
    questions_path: Path,
    views_path: Path,
    max_questions: int | None,
) -> list[WorkItem]:
    question_rows = read_csv(
        questions_path,
        {
            "pilot_id",
            "dataset",
            "split",
            "video_id",
            "video_relative_stem",
            "qid",
            "question_type",
            "duration_seconds",
            "fps",
            "question",
            "candidate_ids_json",
            "candidate_texts_json",
            "gold_candidate_index",
            "evidence_intervals_json",
        },
    )
    view_rows = read_csv(
        views_path,
        {
            "pilot_id",
            "view_id",
            "contract",
            "generator",
            "frame_budget",
            "evidence_intervals_json",
        },
    )
    if max_questions is not None:
        if max_questions <= 0:
            raise ValueError("--max-questions must be positive")
        question_rows = question_rows[:max_questions]

    questions: dict[str, dict[str, str]] = {}
    for row in question_rows:
        pilot_id = row["pilot_id"].strip()
        if not pilot_id or pilot_id in questions:
            raise ValueError(f"Duplicate or empty pilot_id: {pilot_id!r}")
        candidates = parse_json_list(row["candidate_texts_json"], "candidate_texts_json")
        candidate_ids = parse_json_list(row["candidate_ids_json"], "candidate_ids_json")
        if len(candidates) != 5 or len(candidate_ids) != 5:
            raise ValueError(f"{pilot_id}: expected exactly five candidates")
        questions[pilot_id] = row

    views_by_pilot: dict[str, list[dict[str, str]]] = {
        pilot_id: [] for pilot_id in questions
    }
    seen_keys: set[tuple[str, str]] = set()
    for row in view_rows:
        pilot_id = row["pilot_id"].strip()
        if pilot_id not in questions:
            continue
        key = (pilot_id, row["view_id"].strip())
        if key in seen_keys:
            raise ValueError(f"Duplicate view key: {key}")
        seen_keys.add(key)
        generator = row["generator"].strip()
        if generator not in EXPECTED_GENERATORS:
            raise ValueError(f"{key}: unsupported generator {generator!r}")
        if int(row["frame_budget"]) <= 0:
            raise ValueError(f"{key}: frame_budget must be positive")
        views_by_pilot[pilot_id].append(row)

    items: list[WorkItem] = []
    for pilot_id, question in questions.items():
        rows = views_by_pilot[pilot_id]
        if len(rows) != 4:
            raise ValueError(f"{pilot_id}: expected 4 views, found {len(rows)}")
        items.extend(WorkItem(question, view) for view in rows)
    return items


def normalize_intervals(
    intervals: Sequence[Sequence[float]], duration: float
) -> list[tuple[float, float]]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid duration: {duration}")
    clipped: list[tuple[float, float]] = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError(f"Invalid evidence interval: {interval!r}")
        start, end = float(interval[0]), float(interval[1])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"Non-finite evidence interval: {interval!r}")
        start, end = max(0.0, start), min(duration, end)
        if end > start:
            clipped.append((start, end))
    clipped.sort()
    merged: list[tuple[float, float]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def complement_intervals(
    intervals: Sequence[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in intervals:
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        result.append((cursor, duration))
    return result


def sample_across_intervals(
    intervals: Sequence[tuple[float, float]], budget: int
) -> list[float]:
    if budget <= 0:
        raise ValueError("Frame budget must be positive")
    lengths = [end - start for start, end in intervals]
    total = sum(lengths)
    if total <= 0:
        raise ValueError("The requested temporal region has zero duration")
    positions = [(index + 0.5) * total / budget for index in range(budget)]
    timestamps: list[float] = []
    for position in positions:
        traversed = 0.0
        for (start, end), length in zip(intervals, lengths, strict=True):
            if position <= traversed + length or (start, end) == intervals[-1]:
                timestamps.append(start + min(length, position - traversed))
                break
            traversed += length
    return timestamps


def sample_view_timestamps(
    duration: float,
    budget: int,
    generator: str,
    evidence_intervals: Sequence[Sequence[float]],
) -> list[float]:
    evidence = normalize_intervals(evidence_intervals, duration)
    if generator == "uniform_full_video_16_frames":
        return sample_across_intervals([(0.0, duration)], budget)
    if generator == "uniform_phase_shift_16_frames":
        # A deterministic quarter-cell displacement preserves full-video
        # coverage without wrapping the last timestamp to the beginning.
        return [
            min(
                math.nextafter(duration, 0.0),
                (index + 0.75) * duration / (budget + 0.25),
            )
            for index in range(budget)
        ]
    if generator == "nextgqa_evidence_intervals_16_frames":
        return sample_across_intervals(evidence, budget)
    if generator == "complement_of_nextgqa_evidence_intervals_16_frames":
        return sample_across_intervals(
            complement_intervals(evidence, duration), budget
        )
    raise ValueError(f"Unsupported generator: {generator}")


def timestamps_to_indices(
    timestamps: Sequence[float], fps: float, frame_count_hint: int | None = None
) -> list[int]:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")
    maximum = frame_count_hint - 1 if frame_count_hint and frame_count_hint > 0 else None
    result = []
    for timestamp in timestamps:
        index = max(0, int(round(timestamp * fps)))
        if maximum is not None:
            index = min(maximum, index)
        result.append(index)
    return result


def evenly_sample(
    indices: Sequence[int], count: int, *, phase: bool = False
) -> list[int]:
    """Match the locked Qwen runner's ordered frame selection exactly."""

    if count <= 0:
        raise ValueError("Frame budget must be positive")
    if not indices:
        raise ValueError("Cannot sample an empty frame set")
    size = len(indices)
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
    return [int(indices[position]) for position in positions]


def sample_frame_indices(
    frame_count: int,
    duration_seconds: float,
    view_id: str,
    evidence_intervals: Sequence[Sequence[float]],
    budget: int = 16,
) -> list[int]:
    """Return the same locked frame indices used by the Qwen runner."""

    if frame_count <= 0 or duration_seconds <= 0:
        raise ValueError("Video frame count and duration must be positive")
    all_indices = list(range(frame_count))
    intervals = normalize_intervals(evidence_intervals, duration_seconds)
    if view_id == "base":
        return evenly_sample(all_indices, budget)
    if view_id == "invariant_nuisance":
        return evenly_sample(all_indices, budget, phase=True)
    times = [
        (index + 0.5) * duration_seconds / frame_count
        for index in all_indices
    ]
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


@dataclass(frozen=True)
class VideoProbe:
    frame_count: int
    duration_seconds: float
    fps: float


def probe_video(
    path: Path, fallback_duration: float, fallback_fps: float
) -> VideoProbe:
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


def resolve_video_path(video_root: Path, relative_stem: str) -> Path:
    pure = PurePosixPath(relative_stem.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe video_relative_stem: {relative_stem!r}")
    candidate = video_root.joinpath(*pure.parts).with_suffix(".mp4").resolve()
    root = video_root.resolve()
    if root not in candidate.parents:
        raise ValueError(f"Video escaped video root: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def decode_selected_frames(
    video_path: Path, requested_indices: Sequence[int]
) -> tuple[list[Any], int]:
    import av

    if not requested_indices:
        raise ValueError("No frame indices were requested")
    targets = set(requested_indices)
    found: dict[int, Any] = {}
    last_image = None
    decoded_count = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            decoded_count = index + 1
            last_image = frame.to_image().convert("RGB")
            if index in targets:
                found[index] = last_image.copy()
            if index >= max(targets) and targets.issubset(found):
                break
    if last_image is None:
        raise RuntimeError(f"No video frames decoded from {video_path}")
    # Some MP4 headers overstate their frame count.  Reuse the actual final
    # frame for an out-of-range request and record decoded_count for audit.
    frames = [found.get(index, last_image).copy() for index in requested_indices]
    return frames, decoded_count


def preprocess_frames(frames: Sequence[Any], dtype: Any, device: Any) -> Any:
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(
                (448, 448), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    pixels = torch.stack([transform(frame) for frame in frames])
    return pixels.to(device=device, dtype=dtype, non_blocking=True)


def format_question(question: str, candidate_texts: Sequence[str]) -> str:
    options = "\n".join(
        f"{label}. {text}"
        for label, text in zip(OPTION_LABELS, candidate_texts, strict=True)
    )
    return (
        f"Question: {question.strip()}\n"
        f"Options:\n{options}\n"
        "Answer with exactly one option label (A, B, C, D, or E)."
    )


def build_query(
    model: Any,
    tokenizer: Any,
    question_text: str,
    num_patches_list: Sequence[int],
) -> tuple[str, Any, Any]:
    import torch

    frame_markers = "\n".join(
        f"Frame{index + 1}: <image>" for index in range(len(num_patches_list))
    )
    user_text = f"{frame_markers}\n{question_text}"
    template = copy.deepcopy(model.conv_template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], user_text)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    for patch_count in num_patches_list:
        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * model.num_image_token * patch_count
            + "</img>"
        )
        query = query.replace("<image>", image_tokens, 1)
    if "<image>" in query:
        raise RuntimeError("Not every video frame marker was expanded")
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    tokenized = tokenizer(query, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    image_flags = torch.ones(
        (sum(num_patches_list), 1), dtype=torch.long, device=device
    )
    return query, input_ids, (attention_mask, image_flags)


def label_token_ids(tokenizer: Any, label: str) -> list[int]:
    ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError(f"Candidate label {label!r} has no tokens")
    return [int(value) for value in ids]


def contextual_label_token_ids(
    tokenizer: Any, query: str, prefix_ids: Sequence[int], label: str
) -> list[int]:
    """Tokenize a continuation in context and reject a changed prefix."""

    full_ids = tokenizer(query + label, add_special_tokens=True)["input_ids"]
    full_ids = [int(value) for value in full_ids]
    prefix = [int(value) for value in prefix_ids]
    if full_ids[: len(prefix)] != prefix:
        raise RuntimeError(
            f"Appending label {label!r} changed the locked prompt tokenization"
        )
    continuation = full_ids[len(prefix) :]
    if not continuation:
        raise RuntimeError(f"Candidate label {label!r} has no contextual tokens")
    return continuation


def normalized_probabilities(scores: Sequence[float]) -> list[float]:
    maximum = max(scores)
    weights = [math.exp(value - maximum) for value in scores]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    if not all(math.isfinite(value) for value in probabilities):
        raise RuntimeError("Non-finite normalized candidate probability")
    return probabilities


def _token_log_probabilities(
    logits: Any, prefix_length: int, token_ids: Sequence[int]
) -> list[float]:
    import torch

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return [
        float(log_probs[0, prefix_length - 1 + offset, token_id].cpu())
        for offset, token_id in enumerate(token_ids)
    ]


def score_labels(
    model: Any,
    tokenizer: Any,
    pixel_values: Any,
    input_ids: Any,
    attention_mask: Any,
    image_flags: Any,
    query: str | None = None,
) -> dict[str, Any]:
    import torch

    if query is None:
        token_ids = [label_token_ids(tokenizer, label) for label in OPTION_LABELS]
    else:
        prefix_ids = input_ids[0].detach().cpu().tolist()
        token_ids = [
            contextual_label_token_ids(tokenizer, query, prefix_ids, label)
            for label in OPTION_LABELS
        ]
    prefix_length = int(input_ids.shape[1])
    per_candidate: list[list[float]] = []
    forward_passes = 0
    with torch.inference_mode():
        if all(len(ids) == 1 for ids in token_ids):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                use_cache=False,
                return_dict=True,
            )
            forward_passes = 1
            final = torch.log_softmax(outputs.logits[0, -1].float(), dim=-1)
            per_candidate = [
                [float(final[ids[0]].cpu())] for ids in token_ids
            ]
            del outputs, final
        else:
            # Correct teacher-forced fallback for an unexpected tokenizer.
            # It is slower because InternVL's public forward recomputes vision.
            for ids in token_ids:
                continuation = torch.tensor(
                    [ids], dtype=input_ids.dtype, device=input_ids.device
                )
                full_ids = torch.cat((input_ids, continuation), dim=1)
                extra_mask = torch.ones(
                    (1, len(ids)),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                full_mask = torch.cat((attention_mask, extra_mask), dim=1)
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    image_flags=image_flags,
                    use_cache=False,
                    return_dict=True,
                )
                per_candidate.append(
                    _token_log_probabilities(outputs.logits, prefix_length, ids)
                )
                forward_passes += 1
                del outputs

    sequence_scores = [float(sum(values)) for values in per_candidate]
    mean_scores = [
        score / len(values)
        for score, values in zip(sequence_scores, per_candidate, strict=True)
    ]
    probabilities = normalized_probabilities(sequence_scores)
    return {
        "candidate_labels": list(OPTION_LABELS),
        "candidate_token_ids": token_ids,
        "candidate_token_strings": [
            tokenizer.convert_ids_to_tokens(ids) for ids in token_ids
        ],
        "candidate_token_log_probabilities": per_candidate,
        "candidate_sequence_log_probabilities": sequence_scores,
        "candidate_mean_token_log_probabilities": mean_scores,
        "candidate_probabilities": probabilities,
        "candidate_scoring_forward_passes": forward_passes,
        "all_labels_single_token": all(len(ids) == 1 for ids in token_ids),
    }


def load_model(model_id: str, revision: str) -> tuple[Any, Any, Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if revision != MODEL_REVISION:
        raise ValueError(
            "InternVL revision is frozen; expected "
            f"{MODEL_REVISION}, received {revision}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("InternVL inference requires CUDA")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    ).eval().cuda()
    device = next(model.parameters()).device
    return model, tokenizer, dtype, device


class RecordStore:
    def __init__(
        self, state_dir: Path, output: Path, resume: bool, order: Sequence[tuple[str, str]]
    ) -> None:
        self.state_dir = state_dir
        self.records_dir = state_dir / "records"
        self.output = output
        self.order = {key: index for index, key in enumerate(order)}
        if not resume and (state_dir.exists() or output.exists()):
            raise FileExistsError(
                "Output/state already exists; pass --resume to continue"
            )
        self.records_dir.mkdir(parents=True, exist_ok=True)
        if resume and output.exists():
            self._import_jsonl(output)

    @staticmethod
    def _record_key(record: dict[str, Any]) -> tuple[str, str]:
        return str(record["pilot_id"]), str(record["view_id"])

    def shard_path(self, key: tuple[str, str]) -> Path:
        digest = hashlib.sha256(
            f"{key[0]}\0{key[1]}".encode("utf-8")
        ).hexdigest()
        return self.records_dir / f"{digest}.json"

    def _import_jsonl(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Only a truncated final line is recoverable.
                    if any(rest.strip() for rest in handle):
                        raise ValueError(
                            f"Malformed JSONL at {path}:{line_number}"
                        )
                    break
                key = self._record_key(record)
                shard = self.shard_path(key)
                if not shard.exists():
                    atomic_write_text(
                        shard,
                        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n",
                    )

    def put(self, record: dict[str, Any]) -> None:
        key = self._record_key(record)
        atomic_write_text(
            self.shard_path(key),
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n",
        )

    def records(self) -> list[dict[str, Any]]:
        deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
        for path in self.records_dir.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            deduplicated[self._record_key(record)] = record
        return sorted(
            deduplicated.values(),
            key=lambda record: self.order.get(
                self._record_key(record), len(self.order)
            ),
        )

    def completed(self) -> set[tuple[str, str]]:
        return {
            self._record_key(record)
            for record in self.records()
            if record.get("status") == "ok"
        }

    def checkpoint(self) -> int:
        records = self.records()
        content = "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            for record in records
        )
        atomic_write_text(self.output, content)
        audit = {
            "schema_version": SCHEMA_VERSION,
            "updated_at_utc": utc_now(),
            "record_count": len(records),
            "pass_count": sum(record.get("status") == "ok" for record in records),
            "error_count": sum(record.get("status") == "error" for record in records),
            "output": str(self.output.resolve()),
        }
        atomic_write_text(
            self.state_dir / "checkpoint.json",
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
        )
        return len(records)


def base_record(
    item: WorkItem, model_id: str, revision: str
) -> dict[str, Any]:
    question, view = item.question, item.view
    return {
        "schema_version": SCHEMA_VERSION,
        "status": None,
        "created_at_utc": utc_now(),
        "model_id": model_id,
        "model_revision": revision,
        "resolved_revision": revision,
        "record_key": f"{question['pilot_id']}::{view['view_id']}",
        "pilot_id": question["pilot_id"],
        "dataset": question["dataset"],
        "split": question["split"],
        "video_id": question["video_id"],
        "qid": question["qid"],
        "question_type": question["question_type"],
        "view_id": view["view_id"],
        "contract": view["contract"],
        "generator": view["generator"],
        "frame_budget": int(view["frame_budget"]),
    }


def infer_item(
    item: WorkItem,
    video_root: Path,
    model: Any,
    tokenizer: Any,
    dtype: Any,
    device: Any,
    model_id: str,
    revision: str,
    video_hash_cache: dict[Path, str],
) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    question, view = item.question, item.view
    record = base_record(item, model_id, revision)
    try:
        candidates = [
            str(value)
            for value in parse_json_list(
                question["candidate_texts_json"], "candidate_texts_json"
            )
        ]
        candidate_ids = parse_json_list(
            question["candidate_ids_json"], "candidate_ids_json"
        )
        intervals = parse_json_list(
            view["evidence_intervals_json"], "evidence_intervals_json"
        )
        duration = float(question["duration_seconds"])
        fps = float(question["fps"])
        budget = int(view["frame_budget"])
        video_path = resolve_video_path(
            video_root, question["video_relative_stem"]
        )
        probe = probe_video(video_path, duration, fps)
        indices = sample_frame_indices(
            probe.frame_count,
            probe.duration_seconds,
            view["view_id"],
            intervals,
            budget,
        )
        timestamps = [
            (index + 0.5) * probe.duration_seconds / probe.frame_count
            for index in indices
        ]
        frames, decoded_count = decode_selected_frames(video_path, indices)
        pixels = preprocess_frames(frames, dtype, device)
        prompt = format_question(question["question"], candidates)
        query, input_ids, packed = build_query(
            model, tokenizer, prompt, [1] * len(frames)
        )
        attention_mask, image_flags = packed
        scores = score_labels(
            model,
            tokenizer,
            pixels,
            input_ids,
            attention_mask,
            image_flags,
            query,
        )
        probabilities = scores["candidate_probabilities"]
        predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
        gold = int(question["gold_candidate_index"])
        if video_path not in video_hash_cache:
            video_hash_cache[video_path] = sha256_file(video_path)
        record.update(
            {
                "status": "ok",
                "question": question["question"],
                "candidate_ids": candidate_ids,
                "candidate_texts": candidates,
                "ordered_candidate_ids": candidate_ids,
                "ordered_candidate_labels": list(OPTION_LABELS),
                "ordered_candidate_texts": candidates,
                "gold_candidate_index": gold,
                "predicted_candidate_index": predicted,
                "predicted_candidate_label": OPTION_LABELS[predicted],
                "correct": predicted == gold,
                "evidence_intervals_seconds": intervals,
                "sampled_timestamps_seconds": timestamps,
                "sampled_frame_indices": indices,
                "frame_timestamps_seconds": timestamps,
                "frame_indices": indices,
                "decoded_frame_count_until_stop": decoded_count,
                "video_relative_stem": question["video_relative_stem"],
                "video_sha256": video_hash_cache[video_path],
                "prompt_sha256": sha256_text(prompt),
                "expanded_query_sha256": sha256_text(query),
                "prefix_token_count": int(input_ids.shape[1]),
                "dtype": str(dtype),
                "device": str(device),
                **scores,
                "candidate_sequence_scores": scores[
                    "candidate_sequence_log_probabilities"
                ],
                "selected_candidate": OPTION_LABELS[predicted],
                "scoring_method": (
                    "single_prefix_next_token_logits"
                    if scores["candidate_scoring_forward_passes"] == 1
                    else "teacher_forced_sequence_fallback"
                ),
                "videollm_calls": scores["candidate_scoring_forward_passes"],
            }
        )
        del pixels, input_ids, attention_mask, image_flags
    except Exception as exc:  # each failed view remains auditable
        record.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback": traceback.format_exc(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
    record["elapsed_seconds"] = time.perf_counter() - started
    if torch.cuda.is_available():
        record["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    return record


def create_synthetic_video(path: Path) -> None:
    import av
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.mp4")
    container = av.open(str(temporary), mode="w")
    stream = container.add_stream("mpeg4", rate=8)
    stream.width = stream.height = 224
    stream.pix_fmt = "yuv420p"
    try:
        for index in range(16):
            pixels = np.zeros((224, 224, 3), dtype=np.uint8)
            left = 8 + 11 * index
            pixels[88:136, left : left + 48] = (245, 210, 35)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    os.replace(temporary, path)


def run_synthetic_smoke(
    media: Path,
    output: Path,
    model: Any,
    tokenizer: Any,
    dtype: Any,
    device: Any,
    model_id: str,
    revision: str,
) -> None:
    import torch

    if not media.exists():
        create_synthetic_video(media)
    timestamps = sample_view_timestamps(
        duration=2.0,
        budget=8,
        generator="uniform_full_video_16_frames",
        evidence_intervals=[],
    )
    indices = timestamps_to_indices(timestamps, fps=8.0, frame_count_hint=16)
    frames, decoded_count = decode_selected_frames(media, indices)
    pixels = preprocess_frames(frames, dtype, device)
    prompt = (
        "Question: In which direction does the bright square move?\n"
        "Options:\nA. left to right\nB. right to left\nC. upward\n"
        "D. downward\nE. it does not move\n"
        "Answer with exactly one option label (A, B, C, D, or E)."
    )
    query, input_ids, packed = build_query(
        model, tokenizer, prompt, [1] * len(frames)
    )
    first = score_labels(model, tokenizer, pixels, input_ids, *packed, query)
    second = score_labels(model, tokenizer, pixels, input_ids, *packed, query)
    difference = max(
        abs(left - right)
        for left, right in zip(
            first["candidate_sequence_log_probabilities"],
            second["candidate_sequence_log_probabilities"],
            strict=True,
        )
    )
    status = "PASS" if difference <= 1e-5 else "ERROR"
    record = {
        "schema_version": "sicr.internvl.synthetic_smoke.v1",
        "status": status,
        "created_at_utc": utc_now(),
        "model_id": model_id,
        "resolved_revision": revision,
        "hostname": platform.node(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "media_path": str(media.resolve()),
        "media_sha256": sha256_file(media),
        "sampled_frame_indices": indices,
        "decoded_frame_count_until_stop": decoded_count,
        "prompt_sha256": sha256_text(prompt),
        "expanded_query_sha256": sha256_text(query),
        "deterministic_repeat_max_abs_difference": difference,
        "selected_candidate_label": OPTION_LABELS[
            max(
                range(5),
                key=first["candidate_probabilities"].__getitem__,
            )
        ],
        **first,
    }
    if status != "PASS":
        record["error"] = {
            "type": "DeterminismError",
            "message": f"Repeat difference {difference} exceeds 1e-5",
        }
    atomic_write_text(
        output, json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    del pixels, input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if status != "PASS":
        raise RuntimeError(record["error"]["message"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, help="Locked views CSV")
    parser.add_argument("--questions", type=Path, help="Locked questions CSV")
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--media", type=Path)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args(argv)
    public_values = (
        args.manifest,
        args.questions,
        args.video_root,
        args.output,
        args.state_dir,
    )
    if any(value is not None for value in public_values) and not all(
        value is not None for value in public_values
    ):
        parser.error(
            "Public inference requires --manifest, --questions, --video-root, "
            "--output, and --state-dir"
        )
    if args.synthetic_smoke and (
        args.media is None or args.smoke_output is None
    ):
        parser.error("--synthetic-smoke requires --media and --smoke-output")
    if not args.synthetic_smoke and not all(value is not None for value in public_values):
        parser.error("Select --synthetic-smoke and/or provide public inference inputs")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stop_file is not None and args.stop_file.exists():
        if args.manifest is not None:
            items = load_work_items(
                args.questions.resolve(),
                args.manifest.resolve(),
                args.max_questions,
            )
            store = RecordStore(
                args.state_dir.resolve(),
                args.output.resolve(),
                True,
                [item.key for item in items],
            )
            store.checkpoint()
        print(f"STOP present before model load: {args.stop_file}", flush=True)
        return 75
    model, tokenizer, dtype, device = load_model(args.model_id, args.revision)

    if args.synthetic_smoke:
        run_synthetic_smoke(
            args.media.resolve(),
            args.smoke_output.resolve(),
            model,
            tokenizer,
            dtype,
            device,
            args.model_id,
            args.revision,
        )
        print(f"PASS synthetic smoke: {args.smoke_output}", flush=True)

    if args.manifest is None:
        return 0

    items = load_work_items(
        args.questions.resolve(),
        args.manifest.resolve(),
        args.max_questions,
    )
    store = RecordStore(
        args.state_dir.resolve(),
        args.output.resolve(),
        args.resume,
        [item.key for item in items],
    )
    completed = store.completed()
    video_hash_cache: dict[Path, str] = {}
    processed = 0
    errors = 0
    for item in items:
        if args.stop_file is not None and args.stop_file.exists():
            count = store.checkpoint()
            print(
                f"STOP requested; checkpointed {count} records to {args.output}",
                flush=True,
            )
            return 75
        if item.key in completed:
            continue
        record = infer_item(
            item,
            args.video_root.resolve(),
            model,
            tokenizer,
            dtype,
            device,
            args.model_id,
            args.revision,
            video_hash_cache,
        )
        store.put(record)
        processed += 1
        errors += record["status"] == "error"
        print(
            f"{record['status']} {item.key[0]} {item.key[1]} "
            f"elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if processed % args.checkpoint_every == 0:
            count = store.checkpoint()
            print(f"Checkpoint: {count}/{len(items)} records", flush=True)

    count = store.checkpoint()
    print(
        f"COMPLETE records={count}/{len(items)} new_errors={errors} "
        f"output={args.output}",
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
