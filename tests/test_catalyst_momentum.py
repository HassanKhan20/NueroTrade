"""Tests for catalyst-momentum: signal causality, threshold filter, end-to-end."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.strategies.catalyst_momentum import (
    CatalystMomentumParams,
    rank_universe,
    run_catalyst_momentum_backtest,
    score_symbol,
)


def _bars(start: str, n: int, *, drift: float = 0.0,
          vol: float = 0.01, base_vol_share: float = 1e6,
          spike_idx: int | None = None, spike_x: float = 5.0,
          seed: int = 0, strong_close: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n)
    closes = 50.0 * np.exp(np.cumsum(rets))
    vols = np.full(n, base_vol_share, dtype=float)
    if spike_idx is not None and 0 <= spike_idx < n:
        vols[spike_idx] *= spike_x
    idx = pd.date_range(start, periods=n, freq="B")
    if strong_close:
        return pd.DataFrame({
            "open":  closes * (1 - 0.005),
            "high":  closes * (1 + 0.005),
            "low":   closes * (1 - 0.010),
            "close": closes,
            "volume": vols,
        }, index=idx)
    return pd.DataFrame({
        "open":  closes * (1 + 0.005),
        "high":  closes * (1 + 0.010),
        "low":   closes * (1 - 0.005),
        "close": closes,
        "volume": vols,
    }, index=idx)


def _params_filters_off(**overrides) -> CatalystMomentumParams:
    """v1-equivalent params: all v2 filters disabled."""
    base = dict(
        require_strong_close=False,
        require_above_sma=False,
        require_market_uptrend=False,
        soft_stop_pct=0.0,
    )
    base.update(overrides)
    return CatalystMomentumParams(**base)


# ---- score_symbol ----
def test_score_returns_none_when_history_too_short() -> None:
    df = _bars("2024-01-02", 10)
    s = score_symbol(df, pd.Timestamp("2024-01-15"), _params_filters_off())
    assert s is None


def test_score_higher_with_volume_spike_and_momentum() -> None:
    n = 60
    flat = _bars("2024-01-02", n, drift=0.0, vol=0.005, seed=1)
    spiky = _bars("2024-01-02", n, drift=0.005, vol=0.005,
                   spike_idx=n - 1, spike_x=10.0, seed=1)
    p = _params_filters_off()
    ref = pd.Timestamp("2024-04-01")
    s_flat = score_symbol(flat, ref, p)
    s_spiky = score_symbol(spiky, ref, p)
    assert s_flat is not None and s_spiky is not None
    assert s_spiky[0] > s_flat[0]
    assert s_spiky[1] > s_flat[1]


def test_score_strictly_uses_prior_close() -> None:
    """Spike on the ref date itself must NOT affect the score."""
    n = 60
    df = _bars("2024-01-02", n)
    ref_date = df.index[-1]
    p = _params_filters_off()
    s1 = score_symbol(df, ref_date, p)
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] = 1e6
    df2.iloc[-1, df2.columns.get_loc("volume")] = 1e12
    s2 = score_symbol(df2, ref_date, p)
    assert s1 == s2


# ===== v2 fix tests =====
def test_v2_direction_filter_rejects_weak_close() -> None:
    """Same spike + same momentum, but close near the LOW of the day —
    direction filter must reject it."""
    n = 60
    p = _params_filters_off(require_strong_close=True, strong_close_pct=0.6)
    weak = _bars("2024-01-02", n, drift=0.005, spike_idx=n - 1, spike_x=10.0,
                 seed=2, strong_close=False)
    s = score_symbol(weak, pd.Timestamp("2024-04-01"), p)
    assert s is None


def test_v2_direction_filter_accepts_strong_close() -> None:
    """Same spike + same momentum, but close near the HIGH — accepted."""
    n = 60
    p = _params_filters_off(require_strong_close=True, strong_close_pct=0.6)
    strong = _bars("2024-01-02", n, drift=0.005, spike_idx=n - 1, spike_x=10.0,
                   seed=3, strong_close=True)
    s = score_symbol(strong, pd.Timestamp("2024-04-01"), p)
    assert s is not None


def test_v2_trend_filter_rejects_below_sma() -> None:
    """Below SMA(50) → rejected even with a recent spike + uptrend."""
    n = 250
    # Aggressive monotonic decline; final price ~$15 from $100 start.
    closes = np.linspace(100.0, 15.0, n)
    # Tiny 2% bounce in last 5 days, far below the 50-day SMA.
    closes[-5:] *= 1.02
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "open": closes * 0.995, "high": closes * 1.005,
        "low": closes * 0.99, "close": closes,
        "volume": np.where(np.arange(n) == n - 1, 5e7, 1e6),
    }, index=idx)
    p = _params_filters_off(require_above_sma=True, sma_window=50)
    s = score_symbol(df, pd.Timestamp("2025-01-01"), p)
    assert s is None


def test_v2_trend_filter_accepts_above_sma() -> None:
    """Above SMA(50) — accepted (sanity check that it isn't always None)."""
    n = 250
    closes = np.linspace(15.0, 100.0, n)  # monotonic uptrend
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "open": closes * 0.995, "high": closes * 1.005,
        "low": closes * 0.99, "close": closes,
        "volume": np.where(np.arange(n) == n - 1, 5e7, 1e6),
    }, index=idx)
    p = _params_filters_off(require_above_sma=True, sma_window=50)
    s = score_symbol(df, pd.Timestamp("2025-01-01"), p)
    assert s is not None


def test_v2_soft_stop_fires_when_position_drops() -> None:
    """If a held position closes below entry by > soft_stop_pct, exit early."""
    n = 200
    # Build a single stock that:
    # - has enough history
    # - has a pre-rebalance volume spike + uptrend to qualify
    # - then crashes immediately after entry
    rng = np.random.default_rng(7)
    closes = 50.0 * np.exp(np.cumsum(rng.normal(0.001, 0.005, size=n)))
    # Force a 5-day uptrend just before the rebalance
    closes[100:106] *= 1.05
    # Force a crash AFTER the rebalance entry
    closes[107:] *= 0.85   # -15% crash, well below 8% stop
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.005,
        "low":  closes * 0.99, "close": closes,
        "volume": np.where(np.arange(n) == 105, 5e7, 1e6),
    }, index=idx)
    p = _params_filters_off(
        soft_stop_pct=0.08, top_n=1, hold_days=10, spike_threshold=2.0,
    )
    res = run_catalyst_momentum_backtest(
        {"X": df}, start=date(2024, 4, 1), end=date(2024, 12, 1), params=p,
    )
    # If a stop fired, at least one trade exits with a date BEFORE the
    # full hold window. We just check that the strategy didn't ride
    # the full -15% with no protection: a stopped trade limits the loss.
    if res.trades:
        worst = min(t.pnl_net / (t.shares * t.entry_price) for t in res.trades)
        assert worst > -0.13, "soft stop should cap single-trade loss < 13%"


def test_v2_market_regime_filter_blocks_in_bear() -> None:
    """When SPY is in a downtrend, the regime gate should block all trades."""
    n = 250
    rng = np.random.default_rng(13)
    spy_closes = 400.0 * np.exp(np.cumsum(rng.normal(-0.002, 0.005, size=n)))
    spy_idx = pd.date_range("2024-01-02", periods=n, freq="B")
    spy = pd.DataFrame({
        "open": spy_closes * 0.999, "high": spy_closes * 1.003,
        "low": spy_closes * 0.997, "close": spy_closes,
        "volume": 1e8,
    }, index=spy_idx)
    # A "qualifier" biotech that would trade if regime allowed
    bio = _bars("2024-01-02", n, drift=0.005, spike_idx=n - 1, spike_x=8.0,
                seed=11, strong_close=True)
    p = _params_filters_off(
        require_market_uptrend=True, top_n=1, hold_days=5,
        spike_threshold=2.0,
    )
    res = run_catalyst_momentum_backtest(
        {"BIO": bio, "SPY": spy},
        start=date(2024, 7, 1), end=date(2024, 12, 1), params=p,
    )
    # Every rebalance should be tagged "bearish" (or otherwise empty);
    # no real picks should fire.
    bearish_or_empty = all(
        r.get("regime") == "bearish" or r["picks"] == []
        for r in res.rebalance_log
    )
    assert bearish_or_empty


# ---- rank_universe ----
def test_rank_universe_filters_by_threshold() -> None:
    n = 60
    bars = {
        "SPIKE":  _bars("2024-01-02", n, drift=0.005,
                         spike_idx=n - 1, spike_x=8.0, seed=2),
        "QUIET":  _bars("2024-01-02", n, drift=0.005,
                         spike_idx=None, seed=3),  # no spike
        "DOWN":   _bars("2024-01-02", n, drift=-0.005,
                         spike_idx=n - 1, spike_x=8.0, seed=4),  # spike but neg mom
    }
    ranked = rank_universe(bars, date(2024, 4, 1),
                           _params_filters_off(spike_threshold=3.0))
    syms = [s for s, *_ in ranked]
    assert syms == ["SPIKE"]


def test_rank_universe_empty_when_nothing_qualifies() -> None:
    n = 60
    bars = {f"S{i}": _bars("2024-01-02", n, drift=0.0, seed=i) for i in range(5)}
    ranked = rank_universe(bars, date(2024, 4, 1),
                           _params_filters_off(spike_threshold=3.0))
    assert ranked == []


# ---- end-to-end ----
def test_backtest_no_trades_in_quiet_universe() -> None:
    bars = {f"S{i}": _bars("2024-01-02", 200, drift=0.0, vol=0.005, seed=i)
            for i in range(5)}
    result = run_catalyst_momentum_backtest(
        bars, start=date(2024, 5, 1), end=date(2024, 9, 1),
        params=_params_filters_off(spike_threshold=4.0),
    )
    assert result.n_trades() == 0


def test_backtest_no_lookahead() -> None:
    """Two universes identical up to a point — picks before the divergence
    must match exactly regardless of post-divergence prices."""
    bars_a = {"S": _bars("2024-01-02", 200, drift=0.001,
                          spike_idx=120, spike_x=10.0, seed=42)}
    bars_b = {"S": bars_a["S"].copy()}
    div = pd.Timestamp("2024-08-01")
    bars_b["S"].loc[bars_b["S"].index >= div, "close"] *= 0.3
    p = _params_filters_off(top_n=1, hold_days=5, spike_threshold=2.0)
    res_a = run_catalyst_momentum_backtest(
        bars_a, start=date(2024, 5, 1), end=date(2024, 7, 31), params=p,
    )
    res_b = run_catalyst_momentum_backtest(
        bars_b, start=date(2024, 5, 1), end=date(2024, 7, 31), params=p,
    )
    assert [r["picks"] for r in res_a.rebalance_log] == \
           [r["picks"] for r in res_b.rebalance_log]


def test_backtest_handles_empty_universe() -> None:
    res = run_catalyst_momentum_backtest({}, start=date(2024, 1, 1),
                                          end=date(2024, 6, 30))
    assert res.n_trades() == 0
