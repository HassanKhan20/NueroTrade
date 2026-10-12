"""Tests for the Larry Williams volatility-breakout swing strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.williams_breakout import (
    breakout_levels,
    build_williams_callout,
    scan_williams,
    williams_pct_r,
)


def _bars(n: int, drift: float, seed: int = 0,
          start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.015, size=n)
    closes = start_price * np.exp(np.cumsum(rets))
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.012,
        "low": closes * 0.988, "close": closes,
        "volume": np.full(n, 1e6),
    }, index=idx)


def test_pct_r_in_range():
    df = _bars(50, 0.001, seed=1)
    r = williams_pct_r(df, period=10)
    assert -100.0 <= r <= 0.0


def test_pct_r_oversold_when_close_at_low():
    # Construct bars where the last close is at the period low -> %R near -100
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        "open": [100]*10, "high": [110]*10, "low": [90]*10,
        "close": [105]*9 + [90],  # last close == lowest low
        "volume": [1e6]*10,
    }, index=idx)
    assert williams_pct_r(df, 10) == pytest.approx(-100.0)


def test_pct_r_overbought_when_close_at_high():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        "open": [100]*10, "high": [110]*10, "low": [90]*10,
        "close": [95]*9 + [110],  # last close == highest high
        "volume": [1e6]*10,
    }, index=idx)
    assert williams_pct_r(df, 10) == pytest.approx(0.0)


def test_breakout_trigger_uses_prior_bar_no_lookahead():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame({
        "open":  [100, 101, 102, 103, 104],
        "high":  [102, 103, 104, 105, 106],
        "low":   [ 99, 100, 101, 102, 103],
        "close": [101, 102, 103, 104, 105],
        "volume":[1e6]*5,
    }, index=idx)
    # ref = last bar (2024-01-05). Prior bar = 2024-01-04: close 104, range 105-102=3
    ref = pd.Timestamp("2024-01-05")
    res = breakout_levels(df, ref, k=0.5)
    assert res is not None
    prior_close, prior_range, trigger = res
    assert prior_close == pytest.approx(104.0)
    assert prior_range == pytest.approx(3.0)
    assert trigger == pytest.approx(104.0 + 0.5 * 3.0)  # 105.5


def test_breakout_levels_insufficient_history():
    idx = pd.date_range("2024-01-01", periods=1, freq="B")
    df = pd.DataFrame({"open":[100],"high":[101],"low":[99],
                       "close":[100],"volume":[1e6]}, index=idx)
    assert breakout_levels(df, pd.Timestamp("2024-01-02")) is None


def test_callout_levels_consistent():
    df = _bars(300, 0.0015, seed=3)
    c = build_williams_callout("UP", df, pd.Timestamp("2023-02-01"))
    if c is None:
        pytest.skip("snapshot not computable for this synthetic series")
    # Entry above stop, targets above entry, TP2 > TP1
    assert c.stop_price < c.entry_price
    assert c.take_profit_1 > c.entry_price
    assert c.take_profit_2 > c.take_profit_1
    # Risk-per-share matches entry-stop
    assert c.risk_per_share == pytest.approx(c.entry_price - c.stop_price, rel=1e-9)
    # Stop has >= 2.0 ATR of breathing room (validated wider-stop floor)
    assert c.risk_per_share >= 1.95 * c.atr_14


def test_grade_F_when_not_in_uptrend():
    from src.signals.williams_breakout import _grade_williams
    # Strong breakout but NOT in uptrend -> long-only filter forces F
    g = _grade_williams(in_uptrend=False, triggered=True, pct_r=-30,
                        range_to_atr=1.5, rr_tp1=3.0)
    assert g == "F"


def test_grade_F_when_not_triggered():
    from src.signals.williams_breakout import _grade_williams
    # In uptrend but breakout not triggered -> F (pending, not actionable)
    g = _grade_williams(in_uptrend=True, triggered=False, pct_r=-30,
                        range_to_atr=1.5, rr_tp1=3.0)
    assert g == "F"


def test_grade_A_when_all_aligned():
    from src.signals.williams_breakout import _grade_williams
    # Grade A now REQUIRES volume confirmation (>= 1.5x avg)
    g = _grade_williams(in_uptrend=True, triggered=True, pct_r=-35,
                        range_to_atr=1.5, rr_tp1=2.2, rel_volume=2.0)
    assert g == "A"


def test_grade_demoted_without_volume_confirmation():
    from src.signals.williams_breakout import _grade_williams
    # Same strong setup but weak volume -> cannot be A or B
    g = _grade_williams(in_uptrend=True, triggered=True, pct_r=-35,
                        range_to_atr=1.5, rr_tp1=2.2, rel_volume=0.9)
    assert g not in ("A", "B")


def test_scan_excludes_F_and_sorts():
    universe = {f"S{i}": _bars(300, 0.001 + 0.0003*i, seed=i) for i in range(6)}
    callouts = scan_williams(universe, pd.Timestamp("2023-02-01"))
    grades = [c.grade for c in callouts]
    assert "F" not in grades
    # Sorted: grade rank descending
    rank = {"A":5,"B":4,"C":3,"D":2}
    rvals = [rank.get(g, 0) for g in grades]
    assert rvals == sorted(rvals, reverse=True)
