"""Summarize recorded TRACE-RC inference latency without rerunning a model."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(dataset: str, paths: list[Path]) -> list[dict]:
    values: dict[str, list[float]] = defaultdict(list)
    frames: dict[str, int] = defaultdict(int)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                model = str(row["model_id"])
                values[model].append(float(row["latency_seconds"]))
                frames[model] += int(row["input_visual_frames"])
    rows = []
    for model in sorted(values):
        latency = np.asarray(values[model], dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "model_id": model,
                "successful_calls": int(latency.size),
                "input_frames": frames[model],
                "sum_recorded_latency_hours": float(latency.sum() / 3600.0),
                "median_call_latency_seconds": float(np.median(latency)),
                "p95_call_latency_seconds": float(np.percentile(latency, 95)),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--herbench", type=Path, nargs="+")
    parser.add_argument("--perception", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("results/runtime_summary.json"))
    args = parser.parse_args()
    rows = []
    if args.herbench:
        rows.extend(summarize("HERBench Lite-v2", args.herbench))
    if args.perception:
        rows.extend(summarize("Perception Test valid500", args.perception))
    if not rows:
        parser.error("provide --herbench and/or --perception JSONL files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
