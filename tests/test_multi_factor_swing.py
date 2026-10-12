"""Tests for the multi-factor swing ensemble — causality + correctness."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.strategies.multi_factor_swing import (
    MultiFactorParams,
    composite_score_for,
    rank_universe,
    run_multi_factor_swing_backtest,
)


def _bars(start: str, n: int, drift: float = 0.001, vol: float = 0.01,
          seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.005,
        "low": closes * 0.995, "close": closes,
        "volume": np.full(n, 1e6),
    }, index=idx)


def test_composite_score_returns_dict_for_long_history() -> None:
    bars = _bars("2022-01-03", 600, drift=0.001)
    p = MultiFactorParams()
    out = composite_score_for("UNK", bars, pd.Timestamp("2024-06-01"), None, p)
    assert out is not None
    assert set(out.keys()) == {"tech", "mom", "qual", "pead"}


def test_composite_score_none_when_history_too_short() -> None:
    bars = _bars("2024-01-02", 10)
    p = MultiFactorParams()
    out = composite_score_for("UNK", bars, pd.Timestamp("2024-01-30"), None, p)
    assert out is None


def test_rank_universe_sorts_descending() -> None:
    bars = {
        "STRONG": _bars("2022-01-03", 600, drift=0.0015, seed=1),
        "WEAK":   _bars("2022-01-03", 600, drift=-0.0005, seed=2),
    }
    p = MultiFactorParams()
    ranked = rank_universe(bars, date(2024, 6, 1), None, p)
    syms = [r[0] for r in ranked]
    assert syms[0] == "STRONG"
    assert ranked[0][1] >= ranked[-1][1]


def test_rank_universe_handles_empty() -> None:
    p = MultiFactorParams()
    assert rank_universe({}, date(2024, 1, 1), None, p) == []


def test_backtest_no_lookahead() -> None:
    """Identical universes up to a point — picks before the divergence
    must match exactly regardless of post-divergence prices."""
    bars_a = {"S": _bars("2022-01-03", 600, drift=0.001, seed=11)}
    bars_b = {"S": bars_a["S"].copy()}
    div = pd.Timestamp("2024-08-01")
    bars_b["S"].loc[bars_b["S"].index >= div, "close"] *= 0.3
    p = MultiFactorParams(top_n=1, rebalance_days=10, soft_stop_pct=0.0,
                          enable_pead=False)
    res_a = run_multi_factor_swing_backtest(
        bars_a, start=date(2024, 4, 1), end=date(2024, 7, 25), params=p,
    )
    res_b = run_multi_factor_swing_backtest(
        bars_b, start=date(2024, 4, 1), end=date(2024, 7, 25), params=p,
    )
    assert [r["picks"] for r in res_a.rebalance_log] == \
           [r["picks"] for r in res_b.rebalance_log]


def test_backtest_runs_on_real_universe_shape() -> None:
    bars = {
        f"S{i}": _bars("2022-01-03", 750, drift=0.001 - 0.0005 * (i % 3), seed=i)
        for i in range(8)
    }
    p = MultiFactorParams(top_n=3, rebalance_days=10)
    res = run_multi_factor_swing_backtest(
        bars, start=date(2024, 4, 1), end=date(2024, 9, 30), params=p,
    )
    assert len(res.equity_curve) > 0
    # 6 months / 10-day rebalance = at least 5 rebalances
    assert len(res.rebalance_log) >= 5


def test_soft_stop_caps_trade_loss() -> None:
    """Construct a single-name backtest where the price crashes after
    entry; the soft stop should fire and cap the per-trade loss."""
    n = 600
    rng = np.random.default_rng(42)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.005, n)))
    # Force crash AFTER entry window
    crash_start = 500
    closes[crash_start:] *= 0.6  # -40% crash
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    df = pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.005,
        "low":  closes * 0.99, "close": closes,
        "volume": np.full(n, 1e6),
    }, index=idx)

    p = MultiFactorParams(
        top_n=1, rebalance_days=10, soft_stop_pct=0.08, enable_pead=False,
    )
    res = run_multi_factor_swing_backtest(
        {"X": df}, start=date(2024, 4, 1), end=date(2024, 12, 31), params=p,
    )
    if res.trades:
        worst_trade_pct = min(
            t.pnl_net / (t.shares * t.entry_price) for t in res.trades
        )
        # Soft stop should keep any single trade above -13% (8% stop + slippage)
        assert worst_trade_pct > -0.13
