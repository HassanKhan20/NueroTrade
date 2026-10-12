"""Verify the long-only invariant holds across the codebase."""
from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest

from src.strategies.base import Side, Trade
from src.strategies.intraday_momentum import intraday_momentum_signal
from src.strategies.orb_stocks_in_play import ORBParams, orb_signals
from src.utils.clock import NY_TZ


def test_trade_rejects_short_side() -> None:
    with pytest.raises(ValueError, match="Long-only"):
        Trade(
            symbol="X",
            side=Side.SHORT,
            entry_time=datetime(2025, 6, 9, 9, 35),
            entry_price=100.0,
            stop_price=101.0,
            shares=10,
            exit_time=datetime(2025, 6, 9, 15, 55),
            component="orb",
        )


def test_trade_rejects_stop_at_or_above_entry() -> None:
    with pytest.raises(ValueError, match="Long stop must be strictly below"):
        Trade(
            symbol="X",
            side=Side.LONG,
            entry_time=datetime(2025, 6, 9, 9, 35),
            entry_price=100.0,
            stop_price=100.0,
            shares=10,
            exit_time=datetime(2025, 6, 9, 15, 55),
            component="orb",
        )


def test_trade_rejects_zero_shares() -> None:
    with pytest.raises(ValueError, match="shares"):
        Trade(
            symbol="X",
            side=Side.LONG,
            entry_time=datetime(2025, 6, 9, 9, 35),
            entry_price=100.0,
            stop_price=99.5,
            shares=0,
            exit_time=datetime(2025, 6, 9, 15, 55),
            component="orb",
        )


def test_intraday_momentum_does_not_emit_shorts() -> None:
    """Construct a clear short-side day and confirm nothing is emitted."""
    SESSION = date(2025, 6, 9)
    rows = []
    cur = datetime.combine(SESSION, time(9, 30), tzinfo=NY_TZ)
    end = datetime.combine(SESSION, time(15, 56), tzinfo=NY_TZ)
    while cur < end:
        # Steady downtrend: every minute is 0.05 below the prev close
        elapsed = (cur - datetime.combine(SESSION, time(9, 30), tzinfo=NY_TZ)).total_seconds() // 60
        price = 100.0 - 0.01 * float(elapsed)
        rows.append((cur, price, price + 0.05, price - 0.05, price, 100_000))
        cur += pd.Timedelta(minutes=1)
    bars = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    ).set_index("ts")
    t = intraday_momentum_signal(
        bars, prev_close=100.0, session_date=SESSION, equity=100_000.0, symbol="QQQ"
    )
    assert t is None  # would have been a SHORT before; now no trade


def test_orb_does_not_emit_shorts() -> None:
    """A red OR candle must produce zero trades under the long-only rule."""
    SESSION = date(2025, 6, 9)
    rows = []
    # 20 prior days of vanilla bars for relative-volume history
    d = SESSION
    built = 0
    while built < 20:
        d = d.fromordinal(d.toordinal() - 1)
        if d.weekday() >= 5:
            continue
        for m in range(5):
            ts = datetime.combine(d, time(9, 30 + m), tzinfo=NY_TZ)
            rows.append((ts, 100.0, 100.5, 99.5, 100.0, 1_000_000))
        built += 1
    # Today: red OR candle
    rows.append((datetime.combine(SESSION, time(9, 30), tzinfo=NY_TZ),
                 100.0, 100.4, 98.5, 98.8, 5_000_000))
    for m in range(1, 5):
        rows.append((datetime.combine(SESSION, time(9, 30 + m), tzinfo=NY_TZ),
                     98.8, 99.0, 98.5, 98.8, 0))
    rows.sort(key=lambda r: r[0])
    bars = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    ).set_index("ts")

    daily_idx = pd.bdate_range(end=pd.Timestamp("2025-06-06"), periods=30)
    daily = pd.DataFrame(
        {
            "open": [100.0] * len(daily_idx),
            "high": [101.0] * len(daily_idx),
            "low": [99.0] * len(daily_idx),
            "close": [100.0] * len(daily_idx),
            "volume": [1_000_000] * len(daily_idx),
        },
        index=daily_idx,
    )
    trades = orb_signals(
        {"TEST": bars}, {"TEST": daily}, SESSION, equity=100_000.0,
        params=ORBParams(top_n=1),
    )
    assert trades == []
