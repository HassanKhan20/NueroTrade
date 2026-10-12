"""Tests for the inversion mode and take-profit behavior."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from src.backtest.engine import _simulate_momentum_trade, _simulate_orb_trade
from src.strategies.base import Side, Trade
from src.strategies.intraday_momentum import (
    IntradayMomentumParams,
    intraday_momentum_signal,
)
from src.strategies.orb_stocks_in_play import ORBParams, orb_signals
from src.utils.clock import NY_TZ
from tests.test_orb import _build_1min_day, _build_daily_bars

SESSION = date(2025, 6, 9)


# ---------------------------------------------------------------------------
# Trade dataclass: take-profit invariants
# ---------------------------------------------------------------------------
def test_trade_rejects_tp_at_or_below_entry() -> None:
    with pytest.raises(ValueError, match="take-profit must be strictly above"):
        Trade(
            symbol="X",
            side=Side.LONG,
            entry_time=datetime(2025, 6, 9, 9, 35),
            entry_price=100.0,
            stop_price=99.5,
            take_profit_price=100.0,  # equal, not above
            shares=10,
            exit_time=datetime(2025, 6, 9, 15, 55),
            component="orb",
        )


def test_trade_accepts_tp_above_entry() -> None:
    t = Trade(
        symbol="X",
        side=Side.LONG,
        entry_time=datetime(2025, 6, 9, 9, 35),
        entry_price=100.0,
        stop_price=99.5,
        take_profit_price=101.5,
        shares=10,
        exit_time=datetime(2025, 6, 9, 15, 55),
        component="orb",
    )
    assert t.take_profit_price == 101.5


# ---------------------------------------------------------------------------
# ORB inversion: red OR triggers; green OR is skipped
# ---------------------------------------------------------------------------
def test_inverted_orb_takes_red_skips_green() -> None:
    bars_red = {
        "TEST": _build_1min_day(
            or_open=100, or_high=100.4, or_low=98.5, or_close=98.8,
            or_volume=5_000_000, history_volume_per_day=500_000,
        )
    }
    bars_green = {
        "TEST": _build_1min_day(
            or_open=100, or_high=101.5, or_low=99.8, or_close=101.2,
            or_volume=5_000_000, history_volume_per_day=500_000,
        )
    }
    daily = {"TEST": _build_daily_bars(atr_value=1.0)}
    params_inv = ORBParams(top_n=1, inverted=True, max_leverage=100.0)

    red_trades = orb_signals(bars_red, daily, SESSION, equity=100_000.0, params=params_inv)
    green_trades = orb_signals(bars_green, daily, SESSION, equity=100_000.0, params=params_inv)

    assert len(red_trades) == 1
    assert red_trades[0].side == Side.LONG  # still long
    assert green_trades == []


def test_orb_take_profit_set_when_r_positive() -> None:
    bars = {
        "TEST": _build_1min_day(
            or_open=100, or_high=101.0, or_low=99.5, or_close=100.5,
            or_volume=5_000_000, history_volume_per_day=500_000,
        )
    }
    daily = {"TEST": _build_daily_bars(atr_value=1.0)}
    p = ORBParams(top_n=1, take_profit_r=3.0, max_leverage=100.0)
    trades = orb_signals(bars, daily, SESSION, equity=100_000.0, params=p)
    assert len(trades) == 1
    t = trades[0]
    # stop_dist = 0.10 * ATR(=2) = 0.2; TP = entry + 3 * 0.2 = entry + 0.6
    assert t.take_profit_price == pytest.approx(t.entry_price + 0.6)


def test_orb_take_profit_none_when_r_zero() -> None:
    bars = {
        "TEST": _build_1min_day(
            or_open=100, or_high=101.0, or_low=99.5, or_close=100.5,
            or_volume=5_000_000, history_volume_per_day=500_000,
        )
    }
    daily = {"TEST": _build_daily_bars(atr_value=1.0)}
    trades = orb_signals(bars, daily, SESSION, equity=100_000.0,
                         params=ORBParams(top_n=1, take_profit_r=0.0))
    assert trades[0].take_profit_price is None


# ---------------------------------------------------------------------------
# Engine: take-profit fires before stop within a winning bar
# ---------------------------------------------------------------------------
def _build_orb_day_bars(
    *, entry_price: float, hits_tp: bool, tp: float, stop: float
) -> pd.DataFrame:
    start = datetime.combine(SESSION, time(9, 30), tzinfo=NY_TZ)
    rows = []
    for m in range(5):
        ts = start + timedelta(minutes=m)
        rows.append((ts, 100.0, 100.4, 99.6, 100.0, 1000))
    # break-out bar: punches above entry
    rows.append((start + timedelta(minutes=6),
                 entry_price - 0.05, entry_price + 0.10, entry_price - 0.05, entry_price, 5000))
    if hits_tp:
        # massive up bar that touches TP
        rows.append((start + timedelta(minutes=20),
                     entry_price + 0.10, tp + 0.05, entry_price + 0.05, tp + 0.02, 5000))
        rows.append((start + timedelta(minutes=300),
                     tp, tp + 0.05, tp - 0.05, tp - 0.01, 5000))
    else:
        # holds without hitting TP or stop
        rows.append((start + timedelta(minutes=300),
                     entry_price + 0.05, entry_price + 0.10, entry_price - 0.05, entry_price + 0.03, 5000))
    return pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    ).set_index("ts")


def test_engine_take_profit_fires() -> None:
    entry = 101.0
    stop = 100.5
    tp = 102.0
    bars = _build_orb_day_bars(entry_price=entry, hits_tp=True, tp=tp, stop=stop)
    trade = Trade(
        symbol="X", side=Side.LONG,
        entry_time=datetime.combine(SESSION, time(9, 35), tzinfo=NY_TZ),
        entry_price=entry, stop_price=stop, take_profit_price=tp,
        shares=100,
        exit_time=datetime.combine(SESSION, time(15, 55), tzinfo=NY_TZ),
        component="orb",
    )
    f = _simulate_orb_trade(trade, bars)
    assert f is not None
    assert f.exit_reason == "take_profit"
    assert f.pnl_gross > 0
    assert f.exit_price <= tp  # filled at or below TP after slippage on sell


def test_engine_no_tp_uses_time_exit() -> None:
    entry = 101.0
    stop = 100.5
    tp = 102.0
    bars = _build_orb_day_bars(entry_price=entry, hits_tp=False, tp=tp, stop=stop)
    trade = Trade(
        symbol="X", side=Side.LONG,
        entry_time=datetime.combine(SESSION, time(9, 35), tzinfo=NY_TZ),
        entry_price=entry, stop_price=stop, take_profit_price=tp,
        shares=100,
        exit_time=datetime.combine(SESSION, time(15, 55), tzinfo=NY_TZ),
        component="orb",
    )
    f = _simulate_orb_trade(trade, bars)
    assert f is not None
    assert f.exit_reason == "time"


# ---------------------------------------------------------------------------
# Intraday momentum inversion
# ---------------------------------------------------------------------------
def _build_mom_day(closes: dict[time, float]) -> pd.DataFrame:
    """Replicates the helper from test_intraday_momentum but inline."""
    anchors = sorted(closes.items())
    rows = []
    cur = datetime.combine(SESSION, time(9, 30), tzinfo=NY_TZ)
    end = datetime.combine(SESSION, time(15, 56), tzinfo=NY_TZ)
    while cur < end:
        t = cur.time()
        prior, nxt = anchors[0], anchors[-1]
        for a, b in zip(anchors[:-1], anchors[1:]):
            if a[0] <= t <= b[0]:
                prior, nxt = a, b
                break
        if nxt[0] == prior[0]:
            price = prior[1]
        else:
            span = (datetime.combine(SESSION, nxt[0]) - datetime.combine(SESSION, prior[0])).total_seconds()
            elapsed = (datetime.combine(SESSION, t) - datetime.combine(SESSION, prior[0])).total_seconds()
            frac = elapsed / span if span > 0 else 0.0
            price = prior[1] + frac * (nxt[1] - prior[1])
        rng = price * 0.001
        rows.append((cur, price, price + rng, price - rng, price, 100_000))
        cur += timedelta(minutes=1)
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")


def test_inverted_intraday_momentum_takes_both_negative() -> None:
    bars = _build_mom_day({
        time(9, 30): 100.0,
        time(10, 0): 99.0,
        time(14, 0): 99.0,
        time(14, 30): 98.0,
        time(15, 0): 98.0,
        time(15, 55): 98.0,
    })
    p = IntradayMomentumParams(inverted=True)
    t = intraday_momentum_signal(
        bars, prev_close=100.0, session_date=SESSION,
        equity=100_000.0, symbol="QQQ", params=p,
    )
    assert t is not None
    assert t.side == Side.LONG  # still long


def test_inverted_intraday_momentum_skips_both_positive() -> None:
    bars = _build_mom_day({
        time(9, 30): 100.0,
        time(10, 0): 101.0,
        time(14, 0): 101.0,
        time(14, 30): 102.0,
        time(15, 0): 102.0,
        time(15, 55): 102.0,
    })
    p = IntradayMomentumParams(inverted=True)
    t = intraday_momentum_signal(
        bars, prev_close=100.0, session_date=SESSION,
        equity=100_000.0, symbol="QQQ", params=p,
    )
    assert t is None


def test_intraday_momentum_take_profit_set_when_multiple_positive() -> None:
    bars = _build_mom_day({
        time(9, 30): 100.0, time(10, 0): 101.0,
        time(14, 0): 101.0, time(14, 30): 102.0,
        time(15, 0): 102.0, time(15, 55): 102.0,
    })
    p = IntradayMomentumParams(take_profit_multiple=1.5)
    t = intraday_momentum_signal(
        bars, prev_close=100.0, session_date=SESSION,
        equity=100_000.0, symbol="QQQ", params=p,
    )
    assert t is not None
    assert t.take_profit_price is not None
    assert t.take_profit_price > t.entry_price
