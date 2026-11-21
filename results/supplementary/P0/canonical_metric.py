"""P2+ metric: reuse the paper's exact archived prefix-risk mean and tie rule.

No selection is recomputed here. Split 0 must retain its archived decision lock.
The pilot helper's grouped resampling, seeds and thresholds remain unchanged;
only its metric callable should be supplied this function for new splits.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from evaluate_probe_escalation import aurc  # noqa: E402,F401
