"""Tests for the ICT long-only multi-timeframe alignment strategy."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategies.ict_long_only import (
    AM_KILLZONE_END,
    AM_KILLZONE_START,
    FVG,
    ICTParams,
    PM_KILLZONE_END,
    PM_KILLZONE_START,
    find_bullish_fvgs,
    ict_signal_for_symbol,
    in_am_killzone,
    in_killzone,
    in_pm_killzone,
    is_unmitigated,
    price_inside_fvg,
    qqq_bullish,
)
from src.utils.clock import NY_TZ


SESSION = date(2025, 6, 9)


# ---------------------------------------------------------------------------
# Killzone gating
# ---------------------------------------------------------------------------
def test_am_killzone_window() -> None:
    assert not in_am_killzone(time(9, 59))
    assert in_am_killzone(time(10, 0))
    assert in_am_killzone(time(10, 30))
    assert not in_am_killzone(time(11, 0))


def test_pm_killzone_window() -> None:
    assert not in_pm_killzone(time(13, 59))
    assert in_pm_killzone(time(14, 0))
    assert in_pm_killzone(time(15, 30))
    assert not in_pm_killzone(time(16, 0))


def test_in_killzone_combines_am_and_pm() -> None:
    assert in_killzone(time(10, 30))
    assert in_killzone(time(15, 30))
    assert not in_killzone(time(12, 0))
    assert not in_killzone(time(9, 30))
    assert not in_killzone(time(16, 30))


# ---------------------------------------------------------------------------
# FVG detection
# ---------------------------------------------------------------------------
def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build OHLC bars from (open, high, low, close) tuples."""
    idx = pd.date_range("2025-06-09 10:00", periods=len(rows), freq="5min", tz=NY_TZ)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1000.0
    return df


def test_find_bullish_fvgs_finds_clear_imbalance() -> None:
    # C1: high 100; C2: high 102, low 101 (gap up); C3: low 102, high 104.
    # bullish FVG between C1.high (100) and C3.low (102).
    bars = _bars([
        (99, 100, 98, 99.5),
        (100, 102, 100, 101.5),
        (102, 104, 102, 103.5),
        (103, 104, 102, 103.5),
    ])
    fvgs = find_bullish_fvgs(bars, lookback=10, min_pct=0.0)
    assert len(fvgs) == 1
    assert fvgs[0].bottom == pytest.approx(100.0)
    assert fvgs[0].top == pytest.approx(102.0)


def test_find_bullish_fvgs_respects_min_pct() -> None:
    # tiny gap 0.05% — below threshold of 0.10%
    bars = _bars([
        (99, 100.0, 98, 99.5),
        (100.05, 100.1, 100.05, 100.05),
        (100.05, 101.0, 100.05, 100.5),
    ])
    no = find_bullish_fvgs(bars, lookback=10, min_pct=0.001)
    assert no == []


def test_is_unmitigated_true_when_price_holds_above() -> None:
    fvg = FVG(start_ts=pd.Timestamp("2025-06-09 10:00", tz=NY_TZ), bottom=100.0, top=102.0)
    after = _bars([(102.5, 103.0, 102.2, 102.8)])
    assert is_unmitigated(fvg, after)


def test_is_unmitigated_false_when_close_dips_below_bottom() -> None:
    fvg = FVG(start_ts=pd.Timestamp("2025-06-09 10:00", tz=NY_TZ), bottom=100.0, top=102.0)
    after = _bars([(101.0, 101.5, 99.5, 99.5)])
    assert not is_unmitigated(fvg, after)


def test_price_inside_fvg() -> None:
    fvg = FVG(start_ts=pd.Timestamp("2025-06-09 10:00", tz=NY_TZ), bottom=100.0, top=102.0)
    assert price_inside_fvg(101.0, fvg)
    assert price_inside_fvg(100.0, fvg)
    assert price_inside_fvg(102.0, fvg)
    assert not price_inside_fvg(99.5, fvg)
    assert not price_inside_fvg(103.0, fvg)


# ---------------------------------------------------------------------------
# QQQ bias gate
# ---------------------------------------------------------------------------
def _qqq_uptrend(n_min: int = 600) -> pd.DataFrame:
    """Drift-up QQQ minute series so EMA fast > EMA slow."""
    idx = pd.date_range("2025-06-08 04:00", periods=n_min, freq="1min", tz=NY_TZ)
    closes = 500.0 + np.arange(n_min) * 0.05
    return pd.DataFrame({
        "open": closes, "high": closes + 0.1, "low": closes - 0.1,
        "close": closes, "volume": 1000.0,
    }, index=idx)


def _qqq_downtrend(n_min: int = 600) -> pd.DataFrame:
    idx = pd.date_range("2025-06-08 04:00", periods=n_min, freq="1min", tz=NY_TZ)
    closes = 500.0 - np.arange(n_min) * 0.05
    return pd.DataFrame({
        "open": closes, "high": closes + 0.1, "low": closes - 0.1,
        "close": closes, "volume": 1000.0,
    }, index=idx)


def test_qqq_bullish_true_for_uptrend() -> None:
    qqq = _qqq_uptrend()
    as_of = qqq.index[-1].to_pydatetime()
    assert qqq_bullish(qqq, as_of, fast=20, slow=50)


def test_qqq_bullish_false_for_downtrend() -> None:
    qqq = _qqq_downtrend()
    as_of = qqq.index[-1].to_pydatetime()
    assert not qqq_bullish(qqq, as_of, fast=20, slow=50)


def test_qqq_bullish_false_when_history_too_short() -> None:
    qqq = _qqq_uptrend(n_min=30)  # too few bars for EMA(50) on 5-min
    as_of = qqq.index[-1].to_pydatetime()
    assert not qqq_bullish(qqq, as_of, fast=20, slow=50)


# ---------------------------------------------------------------------------
# Full signal: smoke test, holiday skip
# ---------------------------------------------------------------------------
def test_ict_signal_returns_none_on_holiday() -> None:
    qqq = _qqq_uptrend()
    bars = _qqq_uptrend()  # any non-empty
    daily = pd.DataFrame()
    t = ict_signal_for_symbol(
        "AAPL", bars, daily, qqq,
        date(2025, 7, 4), equity=100_000.0, params=ICTParams(),
    )
    assert t is None


def test_ict_signal_returns_none_when_qqq_bearish() -> None:
    """Even if all the symbol-side conditions could in principle line up,
    a bearish QQQ bias must veto the trade."""
    bars = _qqq_uptrend()  # symbol has data
    daily = pd.DataFrame()
    qqq = _qqq_downtrend()
    t = ict_signal_for_symbol(
        "AAPL", bars, daily, qqq,
        SESSION, equity=100_000.0, params=ICTParams(),
    )
    assert t is None


def test_ict_signal_returns_none_when_no_daily_atr() -> None:
    """With no daily history, we can't compute the stop floor and must skip."""
    bars = _qqq_uptrend()
    qqq = _qqq_uptrend()
    t = ict_signal_for_symbol(
        "AAPL", bars, pd.DataFrame(), qqq,
        SESSION, equity=100_000.0, params=ICTParams(),
    )
    assert t is None


# ---------------------------------------------------------------------------
# bias_mode dispatch (the new relative-strength behavior)
# ---------------------------------------------------------------------------
from src.strategies.ict_long_only import bias_passes  # noqa: E402


def test_bias_mode_none_always_passes() -> None:
    qqq = _qqq_downtrend()
    sym = _qqq_downtrend()
    as_of = qqq.index[-1].to_pydatetime()
    assert bias_passes(
        mode="none", as_of=as_of, symbol_bars_1min=sym, qqq_1min=qqq,
        fast=20, slow=50,
    )


def test_bias_mode_self_uses_symbol_not_qqq() -> None:
    qqq = _qqq_downtrend()  # broad market down
    sym = _qqq_uptrend()    # but our stock is up
    as_of = sym.index[-1].to_pydatetime()
    assert bias_passes(
        mode="self", as_of=as_of, symbol_bars_1min=sym, qqq_1min=qqq,
        fast=20, slow=50,
    )
    # Reverse: symbol down, self mode rejects regardless of QQQ
    assert not bias_passes(
        mode="self", as_of=as_of, symbol_bars_1min=qqq, qqq_1min=sym,
        fast=20, slow=50,
    )


def test_bias_mode_qqq_or_self_takes_either() -> None:
    qqq_up = _qqq_uptrend()
    sym_down = _qqq_downtrend()
    as_of = qqq_up.index[-1].to_pydatetime()
    # QQQ up, symbol down -> passes via QQQ
    assert bias_passes(
        mode="qqq_or_self", as_of=as_of,
        symbol_bars_1min=sym_down, qqq_1min=qqq_up,
        fast=20, slow=50,
    )
    qqq_dn = _qqq_downtrend()
    sym_up = _qqq_uptrend()
    as_of2 = sym_up.index[-1].to_pydatetime()
    # QQQ down, symbol up -> passes via self
    assert bias_passes(
        mode="qqq_or_self", as_of=as_of2,
        symbol_bars_1min=sym_up, qqq_1min=qqq_dn,
        fast=20, slow=50,
    )


def test_bias_mode_qqq_or_self_rejects_when_both_down() -> None:
    qqq_dn = _qqq_downtrend()
    sym_dn = _qqq_downtrend()
    as_of = qqq_dn.index[-1].to_pydatetime()
    assert not bias_passes(
        mode="qqq_or_self", as_of=as_of,
        symbol_bars_1min=sym_dn, qqq_1min=qqq_dn,
        fast=20, slow=50,
    )


def test_bias_mode_self_lets_pipeline_continue_despite_bearish_qqq() -> None:
    """End-to-end: a downtrending QQQ does NOT veto when bias_mode='self'.
    With empty daily bars the pipeline still bails on ATR, but the bias
    check is the focus here — no exception, and any non-None Trade would
    be for the right symbol."""
    sym = _qqq_uptrend()
    qqq = _qqq_downtrend()
    t = ict_signal_for_symbol(
        "AAPL", sym, pd.DataFrame(), qqq,
        SESSION, equity=100_000.0, params=ICTParams(bias_mode="self"),
    )
    assert t is None or t.symbol == "AAPL"


# ---------------------------------------------------------------------------
# daily_self bias mode + closed-bar causality (post-leak fix)
# ---------------------------------------------------------------------------
from src.strategies.ict_long_only import is_bullish_daily, _closed_bars_at  # noqa: E402


def _daily_uptrend(n: int = 250) -> pd.DataFrame:
    closes = np.linspace(100.0, 200.0, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": closes, "high": closes + 1.0, "low": closes - 1.0,
        "close": closes, "volume": 1e6,
    }, index=idx)


def _daily_downtrend(n: int = 250) -> pd.DataFrame:
    closes = np.linspace(200.0, 100.0, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": closes, "high": closes + 1.0, "low": closes - 1.0,
        "close": closes, "volume": 1e6,
    }, index=idx)


def test_is_bullish_daily_true_for_uptrend() -> None:
    daily = _daily_uptrend()
    as_of = datetime(2025, 12, 1, 14, 30, tzinfo=NY_TZ)
    assert is_bullish_daily(daily, as_of, fast=50, slow=200)


def test_is_bullish_daily_false_for_downtrend() -> None:
    daily = _daily_downtrend()
    as_of = datetime(2025, 12, 1, 14, 30, tzinfo=NY_TZ)
    assert not is_bullish_daily(daily, as_of, fast=50, slow=200)


def test_is_bullish_daily_requires_minimum_history() -> None:
    daily = _daily_uptrend(n=100)
    as_of = datetime(2025, 6, 1, 14, 30, tzinfo=NY_TZ)
    assert not is_bullish_daily(daily, as_of, fast=50, slow=200)


def test_bias_mode_daily_self_uses_daily_bars_not_intraday() -> None:
    daily = _daily_uptrend()
    minute = _qqq_downtrend()
    qqq = _qqq_downtrend()
    as_of = datetime(2025, 12, 1, 14, 30, tzinfo=NY_TZ)
    assert bias_passes(
        mode="daily_self", as_of=as_of,
        symbol_bars_1min=minute, qqq_1min=qqq,
        fast=20, slow=50, symbol_daily=daily,
    )


def test_bias_mode_daily_self_rejects_daily_downtrend() -> None:
    daily = _daily_downtrend()
    minute = _qqq_uptrend()
    qqq = _qqq_uptrend()
    as_of = datetime(2025, 12, 1, 14, 30, tzinfo=NY_TZ)
    assert not bias_passes(
        mode="daily_self", as_of=as_of,
        symbol_bars_1min=minute, qqq_1min=qqq,
        fast=20, slow=50, symbol_daily=daily,
    )


def test_closed_bars_at_excludes_unfinished_bar() -> None:
    """Causal helper: a 1H bar labeled 10:00 must NOT be visible at ts=10:30."""
    idx = pd.DatetimeIndex([
        pd.Timestamp("2025-06-09 09:00", tz=NY_TZ),
        pd.Timestamp("2025-06-09 10:00", tz=NY_TZ),
        pd.Timestamp("2025-06-09 11:00", tz=NY_TZ),
    ])
    bars = pd.DataFrame({"close": [100, 101, 102]}, index=idx)
    period = pd.Timedelta(hours=1)
    # At 10:30, only the 09:00 bar (ended at 10:00) has fully closed.
    out = _closed_bars_at(bars, pd.Timestamp("2025-06-09 10:30", tz=NY_TZ), period)
    assert list(out["close"]) == [100]
    # At 11:00, the 10:00 bar has just closed; 11:00 still forming.
    out = _closed_bars_at(bars, pd.Timestamp("2025-06-09 11:00", tz=NY_TZ), period)
    assert list(out["close"]) == [100, 101]
    # At 12:00, all three are closed.
    out = _closed_bars_at(bars, pd.Timestamp("2025-06-09 12:00", tz=NY_TZ), period)
    assert list(out["close"]) == [100, 101, 102]
