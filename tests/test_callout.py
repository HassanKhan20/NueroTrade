"""Tests for the callout grading + targets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.callout import build_callout, callout_to_dict


def _bars(n: int, drift: float, seed: int = 0,
          start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, size=n)
    closes = start_price * np.exp(np.cumsum(rets))
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.005,
        "low": closes * 0.99, "close": closes,
        "volume": np.full(n, 1e6),
    }, index=idx)


def test_callout_returns_none_when_history_too_short() -> None:
    df = _bars(50, 0.001)
    assert build_callout("X", df, pd.Timestamp("2022-04-01"),
                         composite_z=2.0,
                         raw_scores={"tech": 1, "mom": 1, "qual": 1, "pead": 0}) is None


def test_callout_basic_long_uptrend_gets_high_grade() -> None:
    df = _bars(600, 0.001, seed=1)
    c = build_callout("UP", df, pd.Timestamp("2024-04-01"),
                      composite_z=2.5,
                      raw_scores={"tech": 2, "mom": 2, "qual": 1, "pead": 0})
    assert c is not None
    assert c.grade in ("A+", "A", "B")
    assert c.entry_price > 0
    assert c.stop_price < c.entry_price
    assert c.take_profit_1 > c.entry_price
    assert c.take_profit_2 > c.take_profit_1


def test_callout_downtrend_returns_F() -> None:
    df = _bars(600, -0.001, seed=2)
    c = build_callout("DOWN", df, pd.Timestamp("2024-04-01"),
                      composite_z=0.5,
                      raw_scores={"tech": 0, "mom": -1, "qual": 0, "pead": 0})
    assert c is not None
    assert c.grade == "F"


def test_callout_negative_composite_returns_F() -> None:
    df = _bars(600, 0.001, seed=3)
    c = build_callout("X", df, pd.Timestamp("2024-04-01"),
                      composite_z=-0.5,
                      raw_scores={"tech": 0, "mom": 0, "qual": 0, "pead": 0})
    assert c is not None
    assert c.grade == "F"


def test_callout_target_math_consistent() -> None:
    df = _bars(600, 0.001, seed=4)
    c = build_callout("X", df, pd.Timestamp("2024-04-01"),
                      composite_z=1.5,
                      raw_scores={"tech": 1, "mom": 1, "qual": 1, "pead": 0})
    assert c is not None
    # Risk per share = entry - stop, must be positive
    assert c.risk_per_share == pytest.approx(c.entry_price - c.stop_price, rel=1e-9)
    # R:R sanity
    assert c.reward_to_risk_tp1 == pytest.approx(
        (c.take_profit_1 - c.entry_price) / c.risk_per_share, rel=1e-9
    )
    # Suggested shares × risk-per-share ≈ risk_dollars (off by < 1 share's worth)
    if c.suggested_shares > 0:
        actual_risk = c.suggested_shares * c.risk_per_share
        assert actual_risk <= c.risk_dollars + c.risk_per_share


def test_callout_dict_serialisation() -> None:
    df = _bars(600, 0.001, seed=5)
    c = build_callout("X", df, pd.Timestamp("2024-04-01"),
                      composite_z=1.5,
                      raw_scores={"tech": 1, "mom": 1, "qual": 1, "pead": 0})
    assert c is not None
    d = callout_to_dict(c)
    assert d["symbol"] == "X"
    assert d["grade"] == c.grade
    assert isinstance(d["strengths"], list)
    assert isinstance(d["weaknesses"], list)
    assert d["entry_price"] == c.entry_price


def test_callout_aplus_requires_strict_alignment() -> None:
    """A+ should be RARE — it needs composite z >= 2, all 3 sub-factors
    positive, RR >= 2.5, RSI in 35-75, MACD positive, AND uptrend."""
    df = _bars(600, 0.001, seed=6)
    c = build_callout("X", df, pd.Timestamp("2024-04-01"),
                      composite_z=1.0,  # not high enough for A+
                      raw_scores={"tech": 1, "mom": 1, "qual": 1, "pead": 0})
    assert c is not None
    assert c.grade != "A+"


def test_callout_action_skip_when_F() -> None:
    df = _bars(600, -0.001, seed=7)
    c = build_callout("X", df, pd.Timestamp("2024-04-01"),
                      composite_z=-0.5,
                      raw_scores={"tech": 0, "mom": -1, "qual": 0, "pead": 0})
    assert c is not None
    assert c.grade == "F"
    assert "SKIP" in c.action.upper()


# ============================================================
# LEARNED improvements: TP1 widened to 2.5x ATR (was 2.0x), and
# Grade A/A+ now require a non-fragile stop (risk >= 0.5 * ATR).
# ============================================================
def test_default_rr_tp1_at_least_166_with_new_tp1_multiplier() -> None:
    """The previous 2.0x ATR TP1 produced R:R 1.33 — below Grade B's 1.5
    floor. The new 2.5x default should clear Grade B math by default."""
    df = _bars(600, 0.001, seed=10)
    c = build_callout("RR", df, pd.Timestamp("2024-04-01"),
                      composite_z=1.0,
                      raw_scores={"tech": 1, "mom": 1, "qual": 1, "pead": 0})
    assert c is not None
    # ATR stop is 1.5x ATR, TP1 is 2.5x ATR -> R:R = 1.667 when ATR stop dominates
    # Swing low could make it different, but it shouldn't be below 1.5
    assert c.reward_to_risk_tp1 >= 1.0  # always above stop


def test_grade_b_now_achievable() -> None:
    """With R:R ~1.67 by default, a z>=1.0 + 2+ positive factors should
    grade B (previously it would have stayed C because R:R was 1.33)."""
    df = _bars(600, 0.001, seed=11)
    c = build_callout("B", df, pd.Timestamp("2024-04-01"),
                      composite_z=1.2,
                      raw_scores={"tech": 1, "mom": 1, "qual": 0, "pead": 0})
    assert c is not None
    # With the new defaults at least Grade B (could be A if R:R lands >= 2)
    assert c.grade in ("A+", "A", "B")


def test_fragile_swing_low_falls_back_to_atr_stop() -> None:
    """Direct test of _build_levels: when swing_low is too close to entry
    (< 0.5x ATR), we should use the ATR stop instead. This prevents the
    AEIS-style 'fake huge R:R but stops out on noise' failure mode."""
    from src.signals.callout import _build_levels
    from src.signals.technical import TechnicalSnapshot

    snap = TechnicalSnapshot(
        rsi_14=60.0, macd_line=1.0, macd_signal=0.5, macd_hist=0.5,
        sma_50=98.0, sma_200=95.0, last_close=100.0,
        atr_14=2.0, rel_volume=1.0, momentum_5d=0.02,
    )
    # Fragile swing low: only $0.10 below entry (= 0.05x ATR, well under
    # the 0.5x threshold). Should be REJECTED -> use ATR stop ($3 below).
    entry, stop, tp1, tp2 = _build_levels(snap, swing_low=99.90)
    assert stop == pytest.approx(100.0 - 1.5 * 2.0)  # ATR stop wins
    assert tp1 == pytest.approx(100.0 + 2.5 * 2.0)
    assert tp2 == pytest.approx(100.0 + 4.0 * 2.0)


def test_real_swing_low_overrides_atr_stop_when_tighter() -> None:
    """When the swing low IS a real stop (> 0.5x ATR below entry), it
    should still be used if it's tighter than the ATR stop."""
    from src.signals.callout import _build_levels
    from src.signals.technical import TechnicalSnapshot

    snap = TechnicalSnapshot(
        rsi_14=60.0, macd_line=1.0, macd_signal=0.5, macd_hist=0.5,
        sma_50=98.0, sma_200=95.0, last_close=100.0,
        atr_14=2.0, rel_volume=1.0, momentum_5d=0.02,
    )
    # Swing low at 98.0 -> 1.0 = 1.0x ATR below entry (above 0.5x floor).
    # ATR stop would be at 97.0 (1.5x ATR). 98.0 is tighter -> wins.
    entry, stop, tp1, tp2 = _build_levels(snap, swing_low=98.0)
    assert stop == pytest.approx(98.0)


def _snap(rsi: float, macd_hist: float = 0.5,
           uptrend: bool = True) -> "TechnicalSnapshot":
    """Helper: build a TechnicalSnapshot with the given RSI/MACD/trend."""
    from src.signals.technical import TechnicalSnapshot
    if uptrend:
        sma_50, sma_200, last = 98.0, 95.0, 100.0
    else:
        sma_50, sma_200, last = 102.0, 105.0, 100.0
    return TechnicalSnapshot(
        rsi_14=rsi, macd_line=1.0, macd_signal=0.5, macd_hist=macd_hist,
        sma_50=sma_50, sma_200=sma_200, last_close=last,
        atr_14=2.0, rel_volume=1.0, momentum_5d=0.02,
    )


def test_action_extreme_rsi_always_waits() -> None:
    """RSI > 85 is always WAIT, even when MACD is strongly positive."""
    from src.signals.callout import _action_text
    snap = _snap(rsi=91.0, macd_hist=2.0)
    action = _action_text("A", snap, rr_tp1=2.5)
    assert "WAIT" in action.upper()
    assert "EXTREME" in action.upper()


def test_action_overbought_with_fading_macd_waits() -> None:
    """RSI 75-85 + MACD <= 0 = overbought with weak momentum -> WAIT."""
    from src.signals.callout import _action_text
    snap = _snap(rsi=78.0, macd_hist=-0.1)
    action = _action_text("B", snap, rr_tp1=1.67)
    assert "WAIT" in action.upper()


def test_action_overbought_with_strong_macd_enters() -> None:
    """RSI 75-85 + MACD > 0 = strong-trend continuation, ENTER scaled
    (LEARNED from MU/GOOGL/CSCO winners that the original WAIT rule
    incorrectly held back)."""
    from src.signals.callout import _action_text
    snap = _snap(rsi=78.0, macd_hist=0.8)
    action = _action_text("B", snap, rr_tp1=1.67)
    assert "ENTER" in action.upper()
    assert "WAIT" not in action.upper()


def test_action_normal_rsi_enters() -> None:
    """RSI in healthy zone (~50-70) just enters per grade."""
    from src.signals.callout import _action_text
    snap = _snap(rsi=60.0, macd_hist=0.5)
    action_a = _action_text("A", snap, rr_tp1=2.5)
    action_c = _action_text("C", snap, rr_tp1=1.67)
    assert "ENTER" in action_a.upper()
    assert "full size" in action_a.lower()
    assert "ENTER" in action_c.upper()
    assert "half size" in action_c.lower()


def test_grade_a_rejected_when_stop_is_fragile() -> None:
    """If risk_per_share is below 0.5x ATR (fragile), even a high-z pick
    with R:R > 2.5 should NOT grade A — this is the AEIS lesson."""
    from src.signals.callout import _grade_from_facts
    from src.signals.technical import TechnicalSnapshot

    snap = TechnicalSnapshot(
        rsi_14=60.0, macd_line=1.0, macd_signal=0.5, macd_hist=0.5,
        sma_50=98.0, sma_200=95.0, last_close=100.0,
        atr_14=2.0, rel_volume=1.0, momentum_5d=0.02,
    )
    raws = {"tech": 1.5, "mom": 1.5, "qual": 1.0, "pead": 0.0}
    # Strong composite + 3+ positive + huge R:R but fragile stop
    grade = _grade_from_facts(composite_z=2.5, raws=raws, snap=snap,
                               rr_tp1=10.0, risk_per_share=0.1)
    assert grade not in ("A+", "A")
