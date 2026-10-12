from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.vol_regime import VolRegimeParams, regime_multiplier


def _vix_series(n: int, value: float) -> pd.Series:
    return pd.Series([value] * n)


def _spy_series(n: int, drift: float = 0.0005, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(0)
    rets = rng.normal(drift, vol, size=n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    return pd.Series(prices)


def test_low_vix_low_vol_returns_full_size() -> None:
    vix_today = 14.0
    vix_hist = _vix_series(252, 18.0)
    spy = _spy_series(252 * 6 + 50)
    m = regime_multiplier(vix_today, vix_hist, spy)
    assert m == 1.0


def test_high_vix_pct_halves_size() -> None:
    # Construct VIX history so today is in the upper tail but not >97pct
    hist = pd.Series(np.linspace(10.0, 30.0, 252))
    today = 28.5  # ~92nd percentile
    spy = _spy_series(252 * 6 + 50)
    m = regime_multiplier(today, hist, spy)
    assert m == 0.5


def test_extreme_vix_halts() -> None:
    hist = pd.Series(np.linspace(10.0, 30.0, 252))
    today = 35.0  # > max -> percentile rank 1.0
    spy = _spy_series(252 * 6 + 50)
    m = regime_multiplier(today, hist, spy)
    assert m == 0.0


def test_nonfinite_vix_halts() -> None:
    m = regime_multiplier(
        float("nan"), _vix_series(252, 18.0), _spy_series(252 * 6 + 50)
    )
    assert m == 0.0


def test_short_history_allows_trading() -> None:
    # Less than 252 days of VIX history -> default to full size
    hist = _vix_series(50, 18.0)
    spy = _spy_series(50)
    m = regime_multiplier(20.0, hist, spy)
    assert m == 1.0
