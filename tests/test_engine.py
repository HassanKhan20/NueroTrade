from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestConfig,
    Backtester,
    DayInputs,
    _simulate_momentum_trade,
    _simulate_orb_trade,
)
from src.strategies.base import Side, Trade
from src.utils.clock import NY_TZ


SESSION = date(2025, 6, 9)


def _bars_with_high_break(entry_price: float, stop_price: float, *, hit_stop: bool) -> pd.DataFrame:
    """Build a 1-min day where price first breaks the entry, then either stops out or holds."""
    start = datetime.combine(SESSION, time(9, 30), tzinfo=NY_TZ)
    rows = []
    # Pre-entry, low volume, prices 100
    for m in range(5):
        ts = start + timedelta(minutes=m)
        rows.append((ts, 100.0, 100.5, 99.5, 100.0, 1000))
    # 09:35 - break above entry_price
    ts = start + timedelta(minutes=6)
    rows.append((ts, entry_price - 0.05, entry_price + 0.10, entry_price - 0.05, entry_price, 5000))
    if hit_stop:
        # later bar dives below stop
        ts2 = start + timedelta(minutes=20)
        rows.append((ts2, stop_price + 0.05, stop_price + 0.05, stop_price - 0.05, stop_price, 5000))
        # eod bar
        ts3 = start + timedelta(minutes=300)
        rows.append((ts3, stop_price - 0.10, stop_price + 0.05, stop_price - 0.20, stop_price - 0.05, 5000))
    else:
        # holds higher till EOD
        ts2 = start + timedelta(minutes=300)
        rows.append((ts2, entry_price + 0.50, entry_price + 0.60, entry_price + 0.40, entry_price + 0.55, 5000))

    return pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    ).set_index("ts")


def test_orb_trade_takes_profit_path() -> None:
    bars = _bars_with_high_break(entry_price=101.0, stop_price=100.5, hit_stop=False)
    trade = Trade(
        symbol="X",
        side=Side.LONG,
        entry_time=datetime.combine(SESSION, time(9, 35), tzinfo=NY_TZ),
        entry_price=101.0,
        stop_price=100.5,
        shares=100,
        exit_time=datetime.combine(SESSION, time(15, 55), tzinfo=NY_TZ),
        component="orb",
    )
    f = _simulate_orb_trade(trade, bars)
    assert f is not None
    assert f.exit_reason == "time"
    assert f.pnl_gross > 0


def test_orb_trade_stop_out_path() -> None:
    bars = _bars_with_high_break(entry_price=101.0, stop_price=100.5, hit_stop=True)
    trade = Trade(
        symbol="X",
        side=Side.LONG,
        entry_time=datetime.combine(SESSION, time(9, 35), tzinfo=NY_TZ),
        entry_price=101.0,
        stop_price=100.5,
        shares=100,
        exit_time=datetime.combine(SESSION, time(15, 55), tzinfo=NY_TZ),
        component="orb",
    )
    f = _simulate_orb_trade(trade, bars)
    assert f is not None
    assert f.exit_reason == "stop"
    assert f.pnl_gross < 0


def test_orb_no_fill_when_high_never_breaks() -> None:
    start = datetime.combine(SESSION, time(9, 30), tzinfo=NY_TZ)
    rows = []
    for m in range(60):
        ts = start + timedelta(minutes=m)
        rows.append((ts, 100.0, 100.4, 99.6, 100.0, 1000))
    bars = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    ).set_index("ts")
    trade = Trade(
        symbol="X",
        side=Side.LONG,
        entry_time=datetime.combine(SESSION, time(9, 35), tzinfo=NY_TZ),
        entry_price=101.0,  # never reached
        stop_price=100.5,
        shares=100,
        exit_time=datetime.combine(SESSION, time(15, 55), tzinfo=NY_TZ),
        component="orb",
    )
    assert _simulate_orb_trade(trade, bars) is None


def test_momentum_trade_pnl_sign_matches_direction() -> None:
    start = datetime.combine(SESSION, time(15, 0), tzinfo=NY_TZ)
    rows = []
    for m in range(120):
        ts = start + timedelta(minutes=m)
        price = 100.0 + 0.01 * m  # uptrending
        rows.append((ts, price, price + 0.02, price - 0.02, price, 1000))
    bars = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    ).set_index("ts")
    trade = Trade(
        symbol="QQQ",
        side=Side.LONG,
        entry_time=start,
        entry_price=100.0,
        stop_price=99.0,
        shares=100,
        exit_time=start + timedelta(minutes=55),
        component="intraday_momentum",
    )
    f = _simulate_momentum_trade(trade, bars)
    assert f is not None
    assert f.pnl_gross > 0


def test_engine_skips_non_trading_days() -> None:
    cfg = BacktestConfig()
    bt = Backtester(cfg)
    # July 4 2025 is a holiday
    holiday = DayInputs(
        session_date=date(2025, 7, 4),
        minute_bars_orb={},
        daily_bars_orb={},
        minute_bars_mom={},
        prev_close_mom={},
        vix_today=18.0,
        vix_history=pd.Series(dtype=float),
        spy_close_history=pd.Series(dtype=float),
    )
    result = bt.run([holiday])
    assert result.n_trades() == 0
    assert result.equity_curve.iloc[0] == cfg.initial_equity
