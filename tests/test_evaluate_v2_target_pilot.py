from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate_v2_target_pilot import aurc, feature_columns, split_environment


def test_aurc_prefers_perfect_ranking() -> None:
    labels = np.asarray([1, 0, 1, 0])
    perfect = np.asarray([0.9, 0.2, 0.8, 0.1])
    reversed_score = -perfect
    assert aurc(labels, perfect) < aurc(labels, reversed_score)


def test_split_keeps_video_groups_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "group_id": np.repeat(["a", "b", "c", "d"], 2),
            "model_id": "m",
            "correct": [0, 1] * 4,
            "own_base_confidence": np.linspace(0.1, 0.8, 8),
        }
    )
    pilot, test = split_environment(frame, 0.5, "fixed")
    assert set(pilot.group_id).isdisjoint(set(test.group_id))
    assert len(pilot) + len(test) == len(frame)


def test_feature_columns_excludes_labels_and_derived_scores() -> None:
    frame = pd.DataFrame(
        {
            "correct": [0, 1],
            "own_base_confidence": [0.2, 0.8],
            "score__raw_confidence": [0.2, 0.8],
            "eco_calls": [1.0, 2.0],
        }
    )
    assert feature_columns(frame) == ["own_base_confidence"]
    assert feature_columns(frame, own_only=True) == ["own_base_confidence"]
