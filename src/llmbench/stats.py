"""Small statistics helpers used by the benchmark report."""

from __future__ import annotations

import math
from statistics import median
from typing import Optional, Sequence


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Return the p-th percentile (0..100) using linear interpolation."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    ordered = sorted(values)
    if p <= 0:
        return float(ordered[0])
    if p >= 100:
        return float(ordered[-1])

    rank = (p / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])

    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def median_or_none(values: Sequence[float]) -> Optional[float]:
    """Return the median of *values*, or ``None`` when the sequence is empty."""
    return float(median(values)) if values else None
