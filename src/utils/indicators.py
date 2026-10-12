"""Pure-numpy technical indicators used across strategies.

Kept dependency-light (numpy only) so the strategy primitives can be unit-
tested without importing pandas.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def true_range(
    high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.float64]:
    """Classic Wilder true range. close is prior close shifted by one day."""
    h = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    if not (h.shape == lo.shape == c.shape):
        raise ValueError("high/low/close must have same shape")
    prev_close = np.concatenate([[c[0]], c[:-1]])
    a = h - lo
    b = np.abs(h - prev_close)
    d = np.abs(lo - prev_close)
    return np.maximum.reduce([a, b, d])


def atr(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, window: int = 14
) -> NDArray[np.float64]:
    """Simple moving average of true range. Returns NaN for the first
    (window-1) bars so callers don't accidentally trade on garbage."""
    if window <= 0:
        raise ValueError("window must be positive")
    tr = true_range(high, low, close)
    n = len(tr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    csum = np.cumsum(tr)
    out[window - 1] = csum[window - 1] / window
    out[window:] = (csum[window:] - csum[:-window]) / window
    return out


def percentile_rank(value: float, history: Sequence[float] | NDArray[np.floating]) -> float:
    """Empirical percentile rank in [0, 1] of value within history.

    Uses 'mean' rule for ties so that constant histories return 0.5.
    """
    h = np.asarray(history, dtype=np.float64)
    if h.size == 0:
        raise ValueError("history must be non-empty")
    less = (h < value).sum()
    eq = (h == value).sum()
    return (less + 0.5 * eq) / h.size


def zscore(value: float, history: Sequence[float] | NDArray[np.floating]) -> float:
    h = np.asarray(history, dtype=np.float64)
    if h.size < 2:
        raise ValueError("history needs at least 2 points for zscore")
    mu = float(h.mean())
    sd = float(h.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return (value - mu) / sd


def realized_vol(returns: ArrayLike, window: int) -> NDArray[np.float64]:
    """Rolling stdev of returns (sample, ddof=1). NaN for first (window-1)."""
    r = np.asarray(returns, dtype=np.float64)
    n = r.size
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2 or n < window:
        return out
    for i in range(window - 1, n):
        out[i] = r[i - window + 1 : i + 1].std(ddof=1)
    return out
