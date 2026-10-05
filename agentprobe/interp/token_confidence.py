"""Optional calibration helpers when a provider returns token log-probabilities."""
from __future__ import annotations

import math
from typing import Iterable


def entropy(log_probs: Iterable[float]) -> float:
    probs = [math.exp(v) for v in log_probs]
    total = sum(probs) or 1.0
    probs = [p / total for p in probs]
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)
