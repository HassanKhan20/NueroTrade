"""Tests for cross-sectional momentum: signal causality, ranking, end-to-end."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.strategies.cross_sectional_momentum import (
    CSMParams,
    _r_12_1,
    rank_universe,
    run_csm_backtest,
)


def _daily_series(start: str, n: int, *, drift: float = 0.0,
                  vol: float = 0.005, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({
        "open": closes * (1 - 0.001),
        "high": closes * (1 + 0.005),
        "low": closes * (1 - 0.005),
        "close": closes,
        "volume": 1e7,
    }, index=idx)


# --- r_12_1 -------------------------------------------------------------
def test_r_12_1_returns_nan_when_history_too_short() -> None:
    df = _daily_series("2024-01-02", 100)
    r = _r_12_1(df["close"], pd.Timestamp("2024-05-01"), lookback=252, skip=21)
    assert np.isnan(r)


def test_r_12_1_positive_for_uptrend() -> None:
    df = _daily_series("2023-01-02", 400, drift=0.001)
    r = _r_12_1(df["close"], pd.Timestamp("2024-06-01"), lookback=252, skip=21)
    assert r > 0


def test_r_12_1_negative_for_downtrend() -> None:
    df = _daily_series("2023-01-02", 400, drift=-0.001)
    r = _r_12_1(df["close"], pd.Timestamp("2024-06-01"), lookback=252, skip=21)
    assert r < 0


def test_r_12_1_strictly_uses_prior_close() -> None:
    """The signal at date D must NOT use D's close — only closes < D."""
    df = _daily_series("2023-01-02", 400)
    # Spike on D and ensure r_12_1 is the same with or without that bar.
    target_date = pd.Timestamp(df.index[-1])
    r1 = _r_12_1(df["close"], target_date, lookback=252, skip=21)
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] = 1e9  # crazy spike on target date
    r2 = _r_12_1(df2["close"], target_date, lookback=252, skip=21)
    assert r1 == r2  # spike on D does not affect signal at D


# --- ranking ------------------------------------------------------------
def test_rank_universe_orders_by_momentum() -> None:
    bars = {
        "UP":   _daily_series("2023-01-02", 400, drift=0.0015, seed=1),
        "FLAT": _daily_series("2023-01-02", 400, drift=0.0,    seed=2),
        "DOWN": _daily_series("2023-01-02", 400, drift=-0.001, seed=3),
    }
    ranked = rank_universe(bars, date(2024, 6, 1), CSMParams())
    syms = [s for s, _ in ranked]
    assert syms[0] == "UP"
    assert syms[-1] == "DOWN"


def test_rank_universe_drops_short_history() -> None:
    bars = {
        "OLD":   _daily_series("2023-01-02", 400, drift=0.001, seed=1),
        "YOUNG": _daily_series("2024-04-01", 30, drift=0.001, seed=2),
    }
    ranked = rank_universe(bars, date(2024, 6, 1), CSMParams())
    syms = [s for s, _ in ranked]
    assert "YOUNG" not in syms
    assert "OLD" in syms


# --- end-to-end ---------------------------------------------------------
def test_csm_backtest_produces_trades_for_winning_universe() -> None:
    """5-symbol universe where 3 of them strongly trend up — strategy
    should pick winners and end positive."""
    bars = {}
    for i, drift in enumerate([0.0015, 0.0012, 0.0010, -0.0005, -0.001]):
        bars[f"S{i}"] = _daily_series("2023-01-02", 400, drift=drift, seed=i)
    p = CSMParams(top_n=2, rebalance_days=21, lookback_days=252, skip_days=21,
                  initial_equity=25_000.0, slippage_bps=2.0)
    result = run_csm_backtest(
        bars, start=date(2024, 1, 5), end=date(2024, 7, 1), params=p,
    )
    assert result.n_trades() > 0
    assert len(result.equity_curve) > 0
    # Strategy should generally pick the strongly trending names
    assert len(result.rebalance_log) >= 5  # ~6 rebalances in 6 months


def test_csm_backtest_handles_empty_universe() -> None:
    p = CSMParams()
    result = run_csm_backtest({}, start=date(2024, 1, 5), end=date(2024, 7, 1), params=p)
    assert result.n_trades() == 0


def test_csm_backtest_no_lookahead_in_signal() -> None:
    """If we feed the strategy two universes that are identical UP TO a
    point, the rebalance picks at that point should be identical regardless
    of any data after that point."""
    bars_a = {"S0": _daily_series("2023-01-02", 400, drift=0.001, seed=10)}
    bars_b = {"S0": bars_a["S0"].copy()}
    bars_b["S0"].loc[bars_b["S0"].index >= "2024-06-02", "close"] *= 0.5
    p = CSMParams(top_n=1, rebalance_days=21)
    res_a = run_csm_backtest(bars_a, start=date(2024, 1, 5), end=date(2024, 5, 30), params=p)
    res_b = run_csm_backtest(bars_b, start=date(2024, 1, 5), end=date(2024, 5, 30), params=p)
    assert [r["picks"] for r in res_a.rebalance_log] == \
           [r["picks"] for r in res_b.rebalance_log]


# ===== v2 recalibration tests =====
def test_abs_momentum_filter_holds_cash_when_all_negative() -> None:
    """Time-series filter at threshold=0: if every name's r_12_1 is
    negative, the strategy should pick zero names (go to cash)."""
    bars = {
        f"D{i}": _daily_series("2023-01-02", 400, drift=-0.0015, seed=i)
        for i in range(5)
    }
    p = CSMParams(
        top_n=3, rebalance_days=21, abs_momentum_threshold=0.0,
        weight_scheme="equal",
    )
    res = run_csm_backtest(
        bars, start=date(2024, 1, 5), end=date(2024, 6, 30), params=p,
    )
    # Every rebalance should have empty picks because all names trend down
    for r in res.rebalance_log:
        assert r["picks"] == [], f"expected cash, got {r['picks']}"
    assert res.n_trades() == 0


def test_abs_momentum_filter_keeps_only_qualifiers() -> None:
    """Mixed universe: only the uptrending names should pass the
    threshold; downtrending ones get filtered even if they're in top-N."""
    bars = {}
    bars["UP"] = _daily_series("2023-01-02", 400, drift=0.0015, seed=1)
    bars["DN1"] = _daily_series("2023-01-02", 400, drift=-0.001, seed=2)
    bars["DN2"] = _daily_series("2023-01-02", 400, drift=-0.0008, seed=3)
    p = CSMParams(
        top_n=3, rebalance_days=21, abs_momentum_threshold=0.0,
        weight_scheme="equal",
    )
    res = run_csm_backtest(
        bars, start=date(2024, 1, 5), end=date(2024, 6, 30), params=p,
    )
    # Only UP should ever appear in picks
    for r in res.rebalance_log:
        for s in r["picks"]:
            assert s == "UP", f"unexpected pick: {s}"


def test_inverse_vol_weights_low_vol_higher() -> None:
    """When weight_scheme=inverse_vol, the lower-vol name should receive a
    larger dollar allocation than the higher-vol name."""
    bars = {
        "QUIET": _daily_series("2023-01-02", 400, drift=0.001, vol=0.003, seed=11),
        "LOUD":  _daily_series("2023-01-02", 400, drift=0.001, vol=0.012, seed=12),
    }
    p_inv = CSMParams(
        top_n=2, rebalance_days=21, abs_momentum_threshold=-1.0,
        weight_scheme="inverse_vol", vol_lookback_days=60,
    )
    p_eq = CSMParams(
        top_n=2, rebalance_days=21, abs_momentum_threshold=-1.0,
        weight_scheme="equal",
    )
    res_inv = run_csm_backtest(
        bars, start=date(2024, 1, 5), end=date(2024, 6, 30), params=p_inv,
    )
    res_eq = run_csm_backtest(
        bars, start=date(2024, 1, 5), end=date(2024, 6, 30), params=p_eq,
    )
    # In inverse-vol mode, total dollars in QUIET > LOUD across the whole run
    inv_quiet = sum(t.shares * t.entry_price for t in res_inv.trades if t.symbol == "QUIET")
    inv_loud  = sum(t.shares * t.entry_price for t in res_inv.trades if t.symbol == "LOUD")
    eq_quiet  = sum(t.shares * t.entry_price for t in res_eq.trades  if t.symbol == "QUIET")
    eq_loud   = sum(t.shares * t.entry_price for t in res_eq.trades  if t.symbol == "LOUD")
    assert inv_quiet > inv_loud, f"inverse-vol: QUIET should outweigh LOUD ({inv_quiet} vs {inv_loud})"
    # Equal-weight should be roughly balanced (both quiet/loud get same dollars)
    if eq_quiet > 0 and eq_loud > 0:
        ratio = eq_quiet / eq_loud
        assert 0.5 < ratio < 2.0, f"equal-weight should be balanced, got ratio {ratio}"
