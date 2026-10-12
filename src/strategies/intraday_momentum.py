"""Component B — Intraday Momentum on SPY/QQQ (LONG-ONLY).

Adapted from Gao, Han, Li, Zhou (JFE 2018). AND-filter variant: enter only
when both the first 30-min return AND the 12th-half-hour return are positive.
Negative-on-both signals are dropped (no shorts in this bot).

  r1  = (close[10:00] - prev_close) / prev_close
  r12 = (close[14:30] - close[14:00]) / close[14:00]
  Long at 15:00 close (exit 15:55) if r1 > 0 AND r12 > 0
  Otherwise: no trade.

Position size: target a fixed dollar volatility,
   shares = (risk_pct * equity) / (atr_5min * sqrt(11))
where the sqrt(11) scales single-bar 5-min ATR to the ~11-bar holding period.

Trade-off vs. long+short: roughly halves the gross signal count and skips
the worst tail-risk regimes (sharp end-of-day rallies on hot CPI/Fed days).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Mapping

import numpy as np
import pandas as pd

from src.strategies.base import Side, Trade
from src.utils.clock import (
    NY_TZ,
    force_flat_time,
    is_trading_day,
)


@dataclass(frozen=True)
class IntradayMomentumParams:
    risk_pct: float = 0.005
    atr_5min_window: int = 14
    use_and_filter: bool = True
    # take_profit_multiple: TP as a multiple of expected_move (atr_5min * sqrt(11)).
    # 0.0 = no profit target. 1.5 = take profit at 1.5 expected moves above entry.
    take_profit_multiple: float = 0.0
    # inverted: long-only mean-reversion variant. Default fires when both
    # r1 > 0 AND r12 > 0 (trend continuation). Inverted fires when both
    # r1 < 0 AND r12 < 0 (close-of-day bounce after a weak session).
    inverted: bool = False


SIGNAL_TIME = time(14, 55)
ENTRY_TIME = time(15, 0)
TENTH_TIME = time(10, 0)
TWELFTH_START_TIME = time(14, 0)
TWELFTH_END_TIME = time(14, 30)


def _close_at(
    bars_1min: pd.DataFrame, session_date: date, t: time
) -> float | None:
    """Close of the 1-min bar whose start equals (session_date, t)."""
    target = datetime.combine(session_date, t, tzinfo=NY_TZ)
    if target not in bars_1min.index:
        # accept the closest bar within +/- 1 minute (handles slightly
        # misaligned vendors)
        nearby = bars_1min.loc[
            (bars_1min.index >= target - pd.Timedelta(minutes=1))
            & (bars_1min.index <= target + pd.Timedelta(minutes=1))
        ]
        if nearby.empty:
            return None
        return float(nearby["close"].iloc[0])
    return float(bars_1min.loc[target, "close"])


def _atr_5min(bars_1min: pd.DataFrame, session_date: date, window: int) -> float:
    """ATR of 5-min bars over the most recent `window` 5-min bars in this session.

    Built from intraday 1-min bars (we resample to 5-min on the fly).
    """
    day_start = datetime.combine(session_date, time(9, 30), tzinfo=NY_TZ)
    day_end = datetime.combine(session_date, time(14, 55), tzinfo=NY_TZ)
    day_bars = bars_1min[(bars_1min.index >= day_start) & (bars_1min.index < day_end)]
    if day_bars.empty:
        return float("nan")
    five = day_bars.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    if len(five) < window:
        return float("nan")
    from src.utils.indicators import atr as _atr

    arr = _atr(
        five["high"].to_numpy(),
        five["low"].to_numpy(),
        five["close"].to_numpy(),
        window=window,
    )
    return float(arr[-1])


def intraday_momentum_signal(
    bars_1min: pd.DataFrame,
    prev_close: float,
    session_date: date,
    equity: float,
    symbol: str,
    params: IntradayMomentumParams = IntradayMomentumParams(),
) -> Trade | None:
    """Return a single Trade or None for one symbol on `session_date`."""
    if not is_trading_day(session_date):
        return None
    if not math.isfinite(prev_close) or prev_close <= 0:
        return None

    c_10 = _close_at(bars_1min, session_date, TENTH_TIME)
    c_14 = _close_at(bars_1min, session_date, TWELFTH_START_TIME)
    c_1430 = _close_at(bars_1min, session_date, TWELFTH_END_TIME)
    c_15 = _close_at(bars_1min, session_date, ENTRY_TIME)
    if None in (c_10, c_14, c_1430, c_15):
        return None
    assert c_10 is not None and c_14 is not None and c_1430 is not None and c_15 is not None

    r1 = (c_10 - prev_close) / prev_close
    r12 = (c_1430 - c_14) / c_14

    # LONG-ONLY entry condition. Default = trend continuation
    # (both half-hour returns positive). Inverted = end-of-day bounce
    # (both negative).
    if params.use_and_filter:
        if params.inverted:
            if not (r1 < 0 and r12 < 0):
                return None
        else:
            if not (r1 > 0 and r12 > 0):
                return None
    else:
        if params.inverted:
            if not (r1 < 0):
                return None
        else:
            if not (r1 > 0):
                return None
    side = Side.LONG

    a5 = _atr_5min(bars_1min, session_date, params.atr_5min_window)
    if not math.isfinite(a5) or a5 <= 0:
        return None
    expected_move = a5 * math.sqrt(11.0)

    risk_dollars = params.risk_pct * equity
    shares = math.floor(risk_dollars / expected_move + 1e-6)
    if shares <= 0:
        return None

    flat_t = force_flat_time(session_date)
    exit_dt = datetime.combine(session_date, flat_t, tzinfo=NY_TZ)
    entry_dt = datetime.combine(session_date, ENTRY_TIME, tzinfo=NY_TZ)
    # No hard stop on this component — the dollar-vol sizing IS the risk
    # control. Notional stop at one expected-move below entry preserves the
    # Trade dataclass invariants.
    stop_price = c_15 - expected_move
    take_profit_price = (
        c_15 + params.take_profit_multiple * expected_move
        if params.take_profit_multiple > 0
        else None
    )

    return Trade(
        symbol=symbol,
        side=side,
        entry_time=entry_dt,
        entry_price=c_15,
        stop_price=stop_price,
        shares=shares,
        exit_time=exit_dt,
        component="intraday_momentum",
        take_profit_price=take_profit_price,
        meta={
            "r1": r1,
            "r12": r12,
            "atr_5min": a5,
            "inverted": float(params.inverted),
        },
    )


def intraday_momentum_signals(
    bars_1min_by_symbol: Mapping[str, pd.DataFrame],
    prev_close_by_symbol: Mapping[str, float],
    session_date: date,
    equity: float,
    params: IntradayMomentumParams = IntradayMomentumParams(),
) -> list[Trade]:
    out: list[Trade] = []
    for sym, bars in bars_1min_by_symbol.items():
        pc = prev_close_by_symbol.get(sym)
        if pc is None:
            continue
        t = intraday_momentum_signal(bars, pc, session_date, equity, sym, params)
        if t is not None:
            out.append(t)
    return out
