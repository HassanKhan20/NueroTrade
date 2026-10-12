from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import Side
from src.strategies.orb_stocks_in_play import (
    ORBParams,
    orb_signals,
    screen_universe,
)
from src.utils.clock import NY_TZ


SESSION = date(2025, 6, 9)  # Monday


def _build_1min_day(
    *,
    or_open: float,
    or_high: float,
    or_low: float,
    or_close: float,
    or_volume: float,
    history_volume_per_day: float,
    history_days: int = 20,
) -> pd.DataFrame:
    """Construct a synthetic 1-min DataFrame: 20 prior trading days with the
    same 09:30-09:35 volume, plus today's OR candle at the configured prices.
    """
    rows = []
    d = SESSION
    days_built = 0
    while days_built < history_days:
        d -= timedelta(days=1)
        if d.weekday() >= 5:
            continue
        for m in range(5):
            t = datetime.combine(d, time(9, 30 + m), tzinfo=NY_TZ)
            rows.append((t, 100.0, 100.5, 99.5, 100.0, history_volume_per_day / 5.0))
        days_built += 1
    # today's OR candle: 5 minute bars 09:30..09:34
    open_t = datetime.combine(SESSION, time(9, 30), tzinfo=NY_TZ)
    rows.append((open_t, or_open, or_high, or_low, or_close, or_volume))
    for m in range(1, 5):
        t = datetime.combine(SESSION, time(9, 30 + m), tzinfo=NY_TZ)
        rows.append((t, or_close, or_high, or_low, or_close, 0.0))

    rows.sort(key=lambda r: r[0])
    df = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    ).set_index("ts")
    return df


def _build_daily_bars(*, atr_value: float = 1.0, days: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp("2025-06-06"), periods=days)
    n = len(idx)
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.0 + atr_value] * n,
            "low": [100.0 - atr_value] * n,
            "close": [100.0] * n,
            "volume": [1_000_000] * n,
        },
        index=idx,
    )


def test_screen_universe_ranks_by_relative_volume() -> None:
    bars = {
        "HIGH": _build_1min_day(
            or_open=100, or_high=101, or_low=99.5, or_close=100.8,
            or_volume=10_000_000, history_volume_per_day=1_000_000,
        ),
        "LOW": _build_1min_day(
            or_open=100, or_high=101, or_low=99.5, or_close=100.8,
            or_volume=500_000, history_volume_per_day=1_000_000,
        ),
    }
    top = screen_universe(bars, SESSION, rel_vol_window=14, top_n=10)
    syms = [s for s, _ in top]
    assert syms == ["HIGH", "LOW"]


def test_green_or_emits_long_at_or_high() -> None:
    bars_1m = {
        "TEST": _build_1min_day(
            or_open=100, or_high=101.5, or_low=99.8, or_close=101.2,
            or_volume=5_000_000, history_volume_per_day=500_000,
        )
    }
    daily = {"TEST": _build_daily_bars(atr_value=1.0)}
    trades = orb_signals(
        bars_1m, daily, SESSION, equity=100_000.0,
        params=ORBParams(top_n=1, atr_window=14, stop_atr_pct=0.10),
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.side == Side.LONG
    assert t.entry_price == pytest.approx(101.5)
    # ATR ≈ 2 (high-low 2 each day), stop = 101.5 - 0.10*2 = 101.3
    assert t.stop_price == pytest.approx(101.3)
    assert t.exit_time.time() == time(15, 55)


def test_red_or_skipped_long_only() -> None:
    bars_1m = {
        "TEST": _build_1min_day(
            or_open=100, or_high=100.4, or_low=98.5, or_close=98.8,
            or_volume=5_000_000, history_volume_per_day=500_000,
        )
    }
    daily = {"TEST": _build_daily_bars(atr_value=1.0)}
    trades = orb_signals(
        bars_1m, daily, SESSION, equity=100_000.0,
        params=ORBParams(top_n=1),
    )
    # Long-only: red OR candles are dropped, no short emitted.
    assert trades == []


def test_doji_skipped() -> None:
    bars_1m = {
        "TEST": _build_1min_day(
            or_open=100, or_high=100.5, or_low=99.5, or_close=100.0,
            or_volume=5_000_000, history_volume_per_day=500_000,
        )
    }
    daily = {"TEST": _build_daily_bars(atr_value=1.0)}
    trades = orb_signals(bars_1m, daily, SESSION, equity=100_000.0)
    assert trades == []


def test_position_sized_to_one_pct_risk() -> None:
    bars_1m = {
        "TEST": _build_1min_day(
            or_open=100, or_high=101.0, or_low=99.5, or_close=100.5,
            or_volume=5_000_000, history_volume_per_day=500_000,
        )
    }
    daily = {"TEST": _build_daily_bars(atr_value=1.0)}
    # Use a large leverage cap so the pure risk-sizing rule is what we test.
    trades = orb_signals(
        bars_1m, daily, SESSION, equity=100_000.0,
        params=ORBParams(top_n=1, risk_per_trade=0.01, max_leverage=100.0),
    )
    assert len(trades) == 1
    # stop_dist = 0.10 * ATR(=2) = 0.2 ; risk $1000 ; shares = 5000
    assert trades[0].shares == 5000
    assert trades[0].risk_dollars() == pytest.approx(1000.0)


def test_no_signal_on_holiday() -> None:
    holiday = date(2025, 7, 4)
    trades = orb_signals({}, {}, holiday, equity=100_000.0)
    assert trades == []


def test_leverage_cap_enforced() -> None:
    # build many candidates so total notional would exceed 4x equity
    bars_1m = {}
    daily = {}
    for i in range(30):
        sym = f"S{i:02d}"
        bars_1m[sym] = _build_1min_day(
            or_open=100, or_high=110.0, or_low=99.5, or_close=109.5,
            or_volume=5_000_000 + i, history_volume_per_day=500_000,
        )
        daily[sym] = _build_daily_bars(atr_value=1.0)
    trades = orb_signals(
        bars_1m, daily, SESSION, equity=10_000.0,
        params=ORBParams(top_n=30, max_leverage=4.0),
    )
    notional = sum(t.shares * t.entry_price for t in trades)
    assert notional <= 4.0 * 10_000.0 + 1e-6
