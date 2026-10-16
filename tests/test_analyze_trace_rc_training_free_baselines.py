from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_trace_rc_training_free_baselines as baselines  # noqa: E402


def example_table() -> pd.DataFrame:
    rows = []
    for qid, role, confidence in (("q1", "test", 0.8), ("q2", "calibration", 0.6)):
        row = {
            "qid": qid,
            "group_id": qid,
            "model_id": "Qwen/Qwen3-VL",
            "correct": 1,
            "role": role,
            "own_base_confidence": confidence,
            "own_base_margin": confidence - 0.2,
            "own_base_entropy": 0.3,
            "base_vote_fraction": 2 / 3,
            "base_selected_support_mean": 0.7,
            "invariant_own_js": 0.1,
            "addition_own_js": 0.2,
            "removal_own_js": 0.3,
        }
        for prefix, vote, support in (
            ("base", 2 / 3, 0.7),
            ("invariant", 1.0, 0.8),
            ("addition", 2 / 3, 0.6),
            ("removal", 1 / 3, 0.5),
        ):
            row[f"{prefix}_vote_fraction"] = vote
            row[f"{prefix}_selected_support_mean"] = support
        rows.append(row)
    return pd.DataFrame(rows)


def test_training_free_scores_use_only_test_rows_and_match_definitions() -> None:
    scores = baselines.training_free_score_frame(example_table())
    assert scores["qid"].tolist() == ["q1"]
    row = scores.iloc[0]
    assert np.isclose(row["score__raw_confidence"], 0.8)
    assert np.isclose(row["score__confidence_margin"], 0.6)
    assert np.isclose(row["score__negative_entropy"], -0.3)
    assert np.isclose(row["score__base_support"], 0.7)
    assert np.isclose(row["score__own_temporal_stability"], -0.2 + 0.8e-9)
    assert np.isclose(row["score__all_view_vote_then_confidence"], 2 / 3 + 0.8e-9)
    assert np.isclose(row["score__all_view_support"], 0.65)


def test_tie_break_keeps_primary_order_and_uses_confidence_for_ties() -> None:
    primary = np.asarray([1.0, 1.0, 2.0])
    confidence = np.asarray([0.4, 0.9, 0.1])
    score = baselines.primary_then_confidence(primary, confidence)
    assert score[1] > score[0]
    assert score[2] > score[1]
