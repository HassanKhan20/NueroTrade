from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import Side
from src.strategies.intraday_momentum import (
    IntradayMomentumParams,
    intraday_momentum_signal,
)
from src.utils.clock import NY_TZ


SESSION = date(2025, 6, 9)


def _build_day(close_curve: dict[time, float], range_pct: float = 0.001) -> pd.DataFrame:
    """Build a 1-min bar series for the full RTH session.

    `close_curve` maps a few key times to absolute closes; remaining bars
    interpolate linearly between adjacent anchors.
    """
    anchors = sorted(close_curve.items())
    rows = []
    cur_t = datetime.combine(SESSION, time(9, 30), tzinfo=NY_TZ)
    end_t = datetime.combine(SESSION, time(15, 56), tzinfo=NY_TZ)
    while cur_t < end_t:
        t = cur_t.time()
        # find bracketing anchors
        prior = anchors[0]
        nxt = anchors[-1]
        for a, b in zip(anchors[:-1], anchors[1:]):
            if a[0] <= t <= b[0]:
                prior, nxt = a, b
                break
        if nxt[0] == prior[0]:
            price = prior[1]
        else:
            span = (
                datetime.combine(SESSION, nxt[0]) - datetime.combine(SESSION, prior[0])
            ).total_seconds()
            elapsed = (
                datetime.combine(SESSION, t) - datetime.combine(SESSION, prior[0])
            ).total_seconds()
            frac = elapsed / span if span > 0 else 0.0
            price = prior[1] + frac * (nxt[1] - prior[1])
        rng = price * range_pct
        rows.append((cur_t, price, price + rng, price - rng, price, 100_000))
        cur_t += timedelta(minutes=1)
    df = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    ).set_index("ts")
    return df


def test_both_positive_emits_long() -> None:
    bars = _build_day(
        {
            time(9, 30): 100.0,
            time(10, 0): 101.0,   # r1 = +1% vs prev_close 100
            time(14, 0): 101.5,
            time(14, 30): 102.0,  # r12 = +0.49%
            time(15, 0): 102.0,
            time(15, 55): 102.0,
        }
    )
    t = intraday_momentum_signal(bars, prev_close=100.0, session_date=SESSION,
                                 equity=100_000.0, symbol="QQQ")
    assert t is not None
    assert t.side == Side.LONG
    assert t.entry_time.time() == time(15, 0)
    assert t.exit_time.time() == time(15, 55)


def test_mixed_signs_no_trade_under_and_filter() -> None:
    bars = _build_day(
        {
            time(9, 30): 100.0,
            time(10, 0): 101.0,    # r1 > 0
            time(14, 0): 101.0,
            time(14, 30): 100.5,   # r12 < 0
            time(15, 0): 100.5,
            time(15, 55): 100.5,
        }
    )
    t = intraday_momentum_signal(bars, prev_close=100.0, session_date=SESSION,
                                 equity=100_000.0, symbol="QQQ")
    assert t is None


def test_both_negative_skipped_long_only() -> None:
    bars = _build_day(
        {
            time(9, 30): 100.0,
            time(10, 0): 99.0,
            time(14, 0): 99.0,
            time(14, 30): 98.0,
            time(15, 0): 98.0,
            time(15, 55): 98.0,
        }
    )
    t = intraday_momentum_signal(bars, prev_close=100.0, session_date=SESSION,
                                 equity=100_000.0, symbol="QQQ")
    # Long-only: negative-on-both signals are dropped, no short emitted.
    assert t is None


def test_size_inverse_to_volatility() -> None:
    # quiet day vs noisy day with the same direction
    quiet = _build_day(
        {
            time(9, 30): 100.0, time(10, 0): 100.5,
            time(14, 0): 100.5, time(14, 30): 100.7,
            time(15, 0): 100.7, time(15, 55): 100.7,
        },
        range_pct=0.0005,
    )
    noisy = _build_day(
        {
            time(9, 30): 100.0, time(10, 0): 100.5,
            time(14, 0): 100.5, time(14, 30): 100.7,
            time(15, 0): 100.7, time(15, 55): 100.7,
        },
        range_pct=0.005,
    )
    a = intraday_momentum_signal(quiet, prev_close=100.0, session_date=SESSION,
                                 equity=100_000.0, symbol="QQQ")
    b = intraday_momentum_signal(noisy, prev_close=100.0, session_date=SESSION,
                                 equity=100_000.0, symbol="QQQ")
    assert a is not None and b is not None
    assert a.shares > b.shares


def test_holiday_no_trade() -> None:
    bars = _build_day({time(9, 30): 100.0, time(15, 55): 101.0})
    t = intraday_momentum_signal(bars, prev_close=100.0,
                                 session_date=date(2025, 7, 4),
                                 equity=100_000.0, symbol="QQQ")
    assert t is None
