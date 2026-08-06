"""Two-phase TRACE-RC analysis for the sealed Perception Test valid500 split."""

from __future__ import annotations

import json

import pandas as pd

import analyze_herbench_trace_rc as analysis


analysis.TARGET_DATASET = "Perception-Test-valid500"
analysis.SEED = 20260803
analysis.PROMOTION_SCHEMA_VERSION = "trace-rc-perception-test-promotion-lock-1.0"
analysis.CONFIRMATION_SCHEMA_VERSION = "trace-rc-perception-test-confirmation-gate-1.0"


def load_manifest(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if len(frame) < 1500 or frame["video_id"].nunique() != 500:
        raise ValueError("unexpected Perception Test valid500 manifest inventory")
    if set(frame["role"]) != {"calibration", "test"}:
        raise ValueError("unexpected Perception Test roles")
    if any(column in frame for column in ("answer", "answer_id", "gold")):
        raise ValueError("label leakage in inference manifest")
    return frame


analysis.load_manifest = load_manifest


if __name__ == "__main__":
    raise SystemExit(analysis.main())
