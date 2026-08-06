from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_trace_rc_posthoc_robustness as audit  # noqa: E402


def test_reported_relative_gain_is_zero_for_identical_scores() -> None:
    labels = np.asarray([1, 0, 1, 0, 1, 0])
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    assert audit.reported_relative_gain(labels, scores, scores) == 0.0


def test_pilot_subsets_are_nested_for_a_fixed_salt() -> None:
    groups = [f"video-{index:03d}" for index in range(50)]
    small = audit.pilot_subset_groups(groups[:25], 50, 0.10, "fixed")
    medium = audit.pilot_subset_groups(groups[:25], 50, 0.30, "fixed")
    full = audit.pilot_subset_groups(groups[:25], 50, 0.50, "fixed")
    assert len(small) == 5
    assert len(medium) == 15
    assert len(full) == 25
    assert small < medium < full


def test_model_family_labels() -> None:
    assert audit.model_family("Qwen/Qwen3-VL") == "Qwen"
    assert audit.model_family("OpenGVLab/InternVL") == "InternVL"
    assert audit.model_family("lmms-lab/LLaVA-OneVision") == "LLaVA"
