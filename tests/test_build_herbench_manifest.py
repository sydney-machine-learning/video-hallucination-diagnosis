from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_herbench_manifest import normalize_choices, strip_option_prefix


def test_strip_option_prefix() -> None:
    assert strip_option_prefix("A. first", 0) == "first"
    assert strip_option_prefix("second", 1) == "second"


def test_normalize_numpy_choices() -> None:
    value = np.asarray(["A. a", "B. b", "C. c", "D. d", "E. e"])
    assert normalize_choices(value) == ["a", "b", "c", "d", "e"]


def test_normalize_four_choices() -> None:
    assert normalize_choices(["A. a", "B. b", "C. c", "D. d"]) == [
        "a",
        "b",
        "c",
        "d",
    ]
