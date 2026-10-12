"""Component A — Stocks-in-Play 5-min Opening Range Breakout (LONG-ONLY).

Rules adapted from Zarattini-Barbon-Aziz, "A Profitable Day Trading Strategy
For The U.S. Equity Market" (SSRN 4729284, 2024), restricted to longs:

  1. At 09:35 ET, rank candidates by relative volume: today's 09:30-09:35
     volume divided by the trailing N-day average of that same 5-min slice.
  2. Take the top K names.
  3. The 09:30-09:35 candle is the opening range. If GREEN (close>open) place
     a buy-stop at OR-high. If red or doji: SKIP (no shorting).
  4. Stop = entry - stop_atr_pct * ATR_14d (in price terms, not %).
  5. Size shares so a stop-out loses risk_per_trade * equity, capped at
     max_leverage * equity total exposure across active orders.
  6. Force-flat at 15:55 ET (12:55 on half days). No profit target.

Note: dropping shorts removes ~half the gross trade count vs. the Zarattini
paper. Expected effect: lower total return, similar Sharpe (long ORB has
historically been the stronger leg in equity bull regimes), substantially
better tail behaviour because squeeze risk on shorts is eliminated.
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
    OR_END,
    OR_START,
    force_flat_time,
    is_trading_day,
)
from src.utils.indicators import atr


@dataclass(frozen=True)
class ORBParams:
    rel_vol_window: int = 14
    top_n: int = 20
    atr_window: int = 14
    stop_atr_pct: float = 0.10
    risk_per_trade: float = 0.01
    max_leverage: float = 4.0
    # take_profit_r: profit target as a multiple of stop distance.
    # 0.0 = no profit target (the original Zarattini-Aziz convention).
    # 3.0 = 3R (caps right tail; reduces upside but improves win rate).
    take_profit_r: float = 0.0
    # inverted: long-only mean-reversion variant. When True, only RED
    # opening-range candles trigger entries — the thesis becomes
    # "morning weakness reverses by close" instead of "morning strength
    # continues by close." Entry/stop/sizing rules are otherwise unchanged.
    inverted: bool = False


def _opening_range_candle(
    bars_1min: pd.DataFrame, session_date: date
) -> tuple[float, float, float, float, float] | None:
    """Aggregate 09:30-09:34 minute bars into one candle.

    Returns (open, high, low, close, volume) or None if data is missing.
    """
    start = datetime.combine(session_date, OR_START, tzinfo=NY_TZ)
    end = datetime.combine(session_date, OR_END, tzinfo=NY_TZ)
    sl = bars_1min[(bars_1min.index >= start) & (bars_1min.index < end)]
    if sl.empty:
        return None
    return (
        float(sl["open"].iloc[0]),
        float(sl["high"].max()),
        float(sl["low"].min()),
        float(sl["close"].iloc[-1]),
        float(sl["volume"].sum()),
    )


def _relative_volume(
    bars_1min: pd.DataFrame, session_date: date, window: int
) -> float:
    """Today's 09:30-09:35 volume / N-day average of same slice. NaN if < window."""
    start = datetime.combine(session_date, OR_START, tzinfo=NY_TZ)
    end = datetime.combine(session_date, OR_END, tzinfo=NY_TZ)
    today_vol = float(
        bars_1min[(bars_1min.index >= start) & (bars_1min.index < end)]["volume"].sum()
    )
    history: list[float] = []
    d = session_date
    while len(history) < window:
        d = d.fromordinal(d.toordinal() - 1)
        if not is_trading_day(d):
            continue
        s = datetime.combine(d, OR_START, tzinfo=NY_TZ)
        e = datetime.combine(d, OR_END, tzinfo=NY_TZ)
        vol = float(
            bars_1min[(bars_1min.index >= s) & (bars_1min.index < e)]["volume"].sum()
        )
        if vol > 0:
            history.append(vol)
        if d.toordinal() < session_date.toordinal() - 365:
            break  # safety: no more than a year of lookback
    if len(history) < window:
        return float("nan")
    avg = sum(history) / len(history)
    if avg == 0:
        return float("nan")
    return today_vol / avg


def _atr_14d(daily_bars: pd.DataFrame, session_date: date, window: int) -> float:
    """ATR of the prior `window` daily bars (does NOT include today)."""
    prior = daily_bars.loc[daily_bars.index < pd.Timestamp(session_date)]
    if len(prior) < window:
        return float("nan")
    arr = atr(
        prior["high"].to_numpy(),
        prior["low"].to_numpy(),
        prior["close"].to_numpy(),
        window=window,
    )
    return float(arr[-1])


def screen_universe(
    bars_1min_by_symbol: Mapping[str, pd.DataFrame],
    session_date: date,
    rel_vol_window: int,
    top_n: int,
) -> list[tuple[str, float]]:
    """Rank symbols by relative volume in the opening 5 minutes; take top N.

    Returns list of (symbol, rel_vol) sorted descending.
    """
    scored: list[tuple[str, float]] = []
    for sym, bars in bars_1min_by_symbol.items():
        rv = _relative_volume(bars, session_date, rel_vol_window)
        if not math.isnan(rv):
            scored.append((sym, rv))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def orb_signals(
    bars_1min_by_symbol: Mapping[str, pd.DataFrame],
    daily_bars_by_symbol: Mapping[str, pd.DataFrame],
    session_date: date,
    equity: float,
    params: ORBParams = ORBParams(),
) -> list[Trade]:
    """Emit Trade intents for one trading session.

    The engine still has to wait for the OR-high/OR-low to be touched intraday;
    this function returns the *resting bracket* the strategy would place at
    09:35:00 ET. Entry price equals the OR break-level — the engine fills it
    when the bar after 09:35 touches it.
    """
    if not is_trading_day(session_date):
        return []

    top = screen_universe(
        bars_1min_by_symbol, session_date, params.rel_vol_window, params.top_n
    )
    if not top:
        return []

    flat_t: time = force_flat_time(session_date)
    exit_dt = datetime.combine(session_date, flat_t, tzinfo=NY_TZ)
    entry_dt = datetime.combine(session_date, OR_END, tzinfo=NY_TZ)

    out: list[Trade] = []
    cumulative_notional = 0.0
    leverage_cap = params.max_leverage * equity
    risk_dollars_per_trade = params.risk_per_trade * equity

    for sym, _rv in top:
        bars = bars_1min_by_symbol.get(sym)
        daily = daily_bars_by_symbol.get(sym)
        if bars is None or daily is None:
            continue
        candle = _opening_range_candle(bars, session_date)
        if candle is None:
            continue
        o, h, lo, c, _v = candle
        a = _atr_14d(daily, session_date, params.atr_window)
        if math.isnan(a) or a <= 0:
            continue

        # LONG-ONLY direction filter.
        # Default (momentum): only GREEN opening-range candles trigger entries.
        # Inverted (reversal): only RED opening-range candles trigger entries
        # (the thesis: morning weakness reverses, so we still buy at OR-high
        # break, just selecting on different setups).
        if params.inverted:
            if c >= o:
                continue
        else:
            if c <= o:
                continue
        side = Side.LONG
        entry_price = h
        stop_price = entry_price - params.stop_atr_pct * a
        take_profit_price = (
            entry_price + params.take_profit_r * (entry_price - stop_price)
            if params.take_profit_r > 0
            else None
        )

        stop_dist = abs(entry_price - stop_price)
        if stop_dist <= 0:
            continue

        # Add a tiny epsilon so 1000.0/0.2 doesn't lose a share to float round-off.
        target_shares = math.floor(risk_dollars_per_trade / stop_dist + 1e-6)
        if target_shares <= 0:
            continue

        notional = target_shares * entry_price
        if cumulative_notional + notional > leverage_cap:
            remaining = leverage_cap - cumulative_notional
            if remaining <= 0:
                break
            target_shares = math.floor(remaining / entry_price + 1e-6)
            if target_shares <= 0:
                continue
            notional = target_shares * entry_price

        cumulative_notional += notional

        out.append(
            Trade(
                symbol=sym,
                side=side,
                entry_time=entry_dt,
                entry_price=entry_price,
                stop_price=stop_price,
                shares=target_shares,
                exit_time=exit_dt,
                component="orb",
                take_profit_price=take_profit_price,
                meta={
                    "or_high": h,
                    "or_low": lo,
                    "atr_14d": a,
                    "inverted": float(params.inverted),
                },
            )
        )
    return out
