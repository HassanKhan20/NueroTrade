"""ICT-style multi-timeframe alignment strategy — long-only equities.

Adapted from "Inner Circle Trader" / Smart Money Concepts. We translate the
discretionary rules into objective, parameterised functions so the bot can
execute mechanically.

Pipeline (all conditions must hold to fire a long):

  1. KILLZONE.   Current NY-time is inside one of the two windows:
                   - NY AM Macro:   10:00 - 11:00 ET ("Silver Bullet")
                   - PM Macro:      14:00 - 16:00 ET
  2. QQQ BIAS.   Long-only mode: only fire if QQQ EMA(20-bar 5-min) >
                 QQQ EMA(50-bar 5-min) at the moment the killzone opens.
                 This is the "bullish bias" gate.
  3. HTF POI.    The symbol's price is currently inside an unmitigated
                 BULLISH 1-HOUR Fair Value Gap (FVG). A bullish 1H FVG is a
                 3-candle imbalance where C1.high < C3.low — the gap is
                 the unfilled wick range [C1.high, C3.low]. "Inside" means
                 the latest 1-min low has dipped below C3.low (price is
                 retracing back into the gap to fill it).
  4. MTF FVG.    A fresh BULLISH 5-MIN FVG has formed on or after the bar
                 that tapped the HTF zone. Same 3-candle definition,
                 5-min timeframe.
  5. LTF ENTRY.  At a 1-min close that exceeds the prior 1-min bar's high
                 (a simple market-structure shift / break-of-structure proxy)
                 — fire a market entry on that close.

Risk:
  - STOP at the lowest 1-min low in the last `stop_lookback` bars
    (default 10) before entry. Capped to a minimum stop distance
    (`min_stop_pct_of_atr` × ATR_14d) so the stop isn't unrealistically tight.
  - SHARES = floor(risk_per_trade * equity / stop_distance).
  - TAKE PROFIT optional via `take_profit_r`; defaults to 2.0 (2R).
  - TIME EXIT at killzone close (11:00 or 16:00) or session 15:55 EOD,
    whichever comes first.

This component intentionally REPLACES the ORB and Intraday Momentum
components when enabled — its cadence is wholly different and mixing them
would be incoherent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Mapping

import numpy as np
import pandas as pd

from src.strategies.base import Side, Trade
from src.utils.clock import NY_TZ, force_flat_time, is_trading_day
from src.utils.indicators import atr


# Killzone definitions (NY time)
AM_KILLZONE_START = time(10, 0)
AM_KILLZONE_END = time(11, 0)
PM_KILLZONE_START = time(14, 0)
PM_KILLZONE_END = time(16, 0)


@dataclass(frozen=True)
class ICTParams:
    # ===== Bias filter =====
    # bias_mode controls which uptrend gate we require before entering a long:
    #   "qqq"          - the index (QQQ) must be in an uptrend.
    #                    Original ICT semantic; can sit dormant for months.
    #   "self"         - the symbol itself must be in an uptrend
    #                    (per-symbol relative strength). Trades any stock
    #                    that is locally bullish even when QQQ is weak.
    #   "spy_or_self"  - trade if EITHER SPY is bullish OR the symbol is.
    #                    More permissive; uses QQQ-style bars for SPY too.
    #   "none"         - no bias filter at all (purely structural).
    bias_mode: str = "qqq"
    bias_ema_fast: int = 20  # 5-min bars
    bias_ema_slow: int = 50
    # legacy aliases retained so existing call sites keep working
    qqq_ema_fast: int = 20
    qqq_ema_slow: int = 50

    # ===== FVG detection =====
    htf_fvg_min_pct: float = 0.001
    mtf_fvg_min_pct: float = 0.0005
    htf_lookback_bars: int = 24
    mtf_lookback_bars: int = 12

    # ===== Risk management =====
    risk_per_trade: float = 0.01
    max_leverage: float = 4.0
    stop_lookback_bars: int = 10
    min_stop_pct_of_atr: float = 0.05
    atr_window_daily: int = 14
    take_profit_r: float = 2.0

    # Filter: only one trade per symbol per session
    one_per_symbol_per_day: bool = True


# --------------------------------------------------------------------------- #
# Killzone helpers
# --------------------------------------------------------------------------- #
def in_am_killzone(t: time) -> bool:
    return AM_KILLZONE_START <= t < AM_KILLZONE_END


def in_pm_killzone(t: time) -> bool:
    return PM_KILLZONE_START <= t < PM_KILLZONE_END


def in_killzone(t: time) -> bool:
    return in_am_killzone(t) or in_pm_killzone(t)


def killzone_end_for(t: time) -> time:
    """Returns the killzone end time that contains `t`, or 15:55 if not."""
    if in_am_killzone(t):
        return AM_KILLZONE_END
    if in_pm_killzone(t):
        return PM_KILLZONE_END
    return time(15, 55)


# --------------------------------------------------------------------------- #
# QQQ bias
# --------------------------------------------------------------------------- #
def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Standard pandas-style EMA."""
    if values.size == 0:
        return values
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def is_bullish_5m(
    bars_1min: pd.DataFrame, as_of: datetime, fast: int, slow: int
) -> bool:
    """True iff the symbol's 5-min EMA(fast) > EMA(slow) at `as_of`.

    CAUSAL: only uses 5-min bars whose period has fully CLOSED by `as_of`.
    """
    if bars_1min.empty:
        return False
    upto = bars_1min[bars_1min.index <= as_of]
    if upto.empty:
        return False
    five = (
        upto.resample("5min", label="left", closed="left")
        .agg({"close": "last"})
        .dropna()
    )
    # Drop any 5-min bar whose end > as_of (i.e. still forming).
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize(NY_TZ)
    five = five[five.index + pd.Timedelta(minutes=5) <= as_of_ts]
    needed = max(fast, slow) + 5
    if len(five) < needed:
        return False
    closes = five["close"].to_numpy(dtype=np.float64)
    ema_f = _ema(closes, fast)
    ema_s = _ema(closes, slow)
    return bool(ema_f[-1] > ema_s[-1])


# Backward-compat alias for the original name; same semantics.
def qqq_bullish(
    qqq_1min: pd.DataFrame, as_of: datetime, fast: int, slow: int
) -> bool:
    return is_bullish_5m(qqq_1min, as_of, fast, slow)


def is_bullish_daily(
    daily_bars: pd.DataFrame, as_of: datetime, fast: int = 50, slow: int = 200
) -> bool:
    """True iff fast-SMA > slow-SMA on the symbol's daily bars at as_of.

    CAUSAL: uses only daily bars whose date is strictly BEFORE as_of's date
    (a daily bar for date D isn't 'closed' until D+1 in NY time).
    Far less restrictive than 5-min EMA — captures durable trends.
    """
    if daily_bars is None or daily_bars.empty:
        return False
    if not isinstance(daily_bars.index, pd.DatetimeIndex):
        return False
    cutoff = (
        pd.Timestamp(as_of).tz_convert(daily_bars.index.tz)
        if (daily_bars.index.tz is not None and pd.Timestamp(as_of).tzinfo is not None)
        else pd.Timestamp(as_of)
    )
    # Strip tz for date comparison if needed
    if daily_bars.index.tz is None and pd.Timestamp(as_of).tzinfo is not None:
        cutoff = pd.Timestamp(as_of).tz_localize(None)
    sess_date = cutoff.date() if hasattr(cutoff, "date") else cutoff
    if daily_bars.index.tz is not None:
        prior = daily_bars[daily_bars.index.date < sess_date]
    else:
        prior = daily_bars[daily_bars.index.date < sess_date]
    if len(prior) < slow + 1:
        return False
    closes = prior["close"].to_numpy(dtype=np.float64)
    sma_f = closes[-fast:].mean()
    sma_s = closes[-slow:].mean()
    return bool(sma_f > sma_s)


def bias_passes(
    *,
    mode: str,
    as_of: datetime,
    symbol_bars_1min: pd.DataFrame,
    qqq_1min: pd.DataFrame,
    fast: int,
    slow: int,
    symbol_daily: pd.DataFrame | None = None,
) -> bool:
    """Dispatch on bias_mode. Unknown modes are treated as 'qqq' (safest).

    Modes:
      qqq           - QQQ 5-min EMA(fast) > EMA(slow)        (original)
      self          - symbol 5-min EMA(fast) > EMA(slow)     (relative strength, intraday)
      qqq_or_self   - either of the above
      daily_self    - symbol DAILY 50-SMA > 200-SMA          (durable trend, far more permissive)
      none          - no filter
    """
    if mode == "none":
        return True
    if mode == "self":
        return is_bullish_5m(symbol_bars_1min, as_of, fast, slow)
    if mode == "qqq_or_self":
        return is_bullish_5m(qqq_1min, as_of, fast, slow) or is_bullish_5m(
            symbol_bars_1min, as_of, fast, slow
        )
    if mode == "daily_self":
        if symbol_daily is None:
            return False
        return is_bullish_daily(symbol_daily, as_of, fast=50, slow=200)
    # default: "qqq"
    return is_bullish_5m(qqq_1min, as_of, fast, slow)


# --------------------------------------------------------------------------- #
# Fair Value Gap detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FVG:
    """A bullish 3-candle imbalance: C1.high < C3.low. The gap zone is
    [C1.high, C3.low]; price is "inside" when low dips below C3.low and
    "filled" once a subsequent close reaches back below C1.high."""

    start_ts: pd.Timestamp
    bottom: float  # C1.high
    top: float  # C3.low

    def width(self) -> float:
        return self.top - self.bottom


def find_bullish_fvgs(
    bars: pd.DataFrame, lookback: int, min_pct: float
) -> list[FVG]:
    """Return all bullish FVGs in the last `lookback` bars whose width is
    >= min_pct of the third candle's close. Returned ordered by start."""
    if bars is None or bars.empty or len(bars) < 3:
        return []
    sl = bars.tail(lookback + 2)  # need 2 extra to form the 3-bar window
    out: list[FVG] = []
    highs = sl["high"].to_numpy()
    lows = sl["low"].to_numpy()
    closes = sl["close"].to_numpy()
    idx = sl.index
    for i in range(2, len(sl)):
        c1_high = highs[i - 2]
        c3_low = lows[i]
        c3_close = closes[i]
        if c1_high < c3_low:
            width = c3_low - c1_high
            if c3_close > 0 and (width / c3_close) >= min_pct:
                out.append(FVG(start_ts=idx[i - 2], bottom=c1_high, top=c3_low))
    return out


def is_unmitigated(fvg: FVG, bars_after: pd.DataFrame) -> bool:
    """An FVG is unmitigated if no subsequent close has fully filled it
    (i.e. no close at or below the FVG's bottom, which would mean price
    traversed the entire gap)."""
    if bars_after.empty:
        return True
    return bool((bars_after["close"] > fvg.bottom).all())


def price_inside_fvg(price: float, fvg: FVG) -> bool:
    return fvg.bottom <= price <= fvg.top


# --------------------------------------------------------------------------- #
# Daily ATR for the stop floor
# --------------------------------------------------------------------------- #
def _atr_14d(daily_bars: pd.DataFrame, session_date: date, window: int) -> float:
    if daily_bars is None or daily_bars.empty:
        return float("nan")
    if not isinstance(daily_bars.index, pd.DatetimeIndex):
        return float("nan")
    cutoff = (
        pd.Timestamp(session_date, tz=daily_bars.index.tz)
        if daily_bars.index.tz is not None
        else pd.Timestamp(session_date)
    )
    prior = daily_bars.loc[daily_bars.index < cutoff]
    if len(prior) < window:
        return float("nan")
    arr = atr(
        prior["high"].to_numpy(),
        prior["low"].to_numpy(),
        prior["close"].to_numpy(),
        window=window,
    )
    return float(arr[-1])


# --------------------------------------------------------------------------- #
# Main per-symbol signal generator
# --------------------------------------------------------------------------- #
def _resample_to(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1-min bars to the requested timeframe."""
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return (
        bars.resample(rule, label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


def _closed_bars_at(bars: pd.DataFrame, ts: pd.Timestamp, period: pd.Timedelta) -> pd.DataFrame:
    """Return only bars whose period has *fully closed* by ts.

    A bar labeled L (left-closed resample) covers [L, L+period). It must
    not be used until ts >= L + period — otherwise we would be using
    future minutes that haven't happened yet relative to the simulation
    clock. This is the canonical multi-timeframe causality fix.
    """
    if bars is None or bars.empty:
        return bars
    return bars[bars.index + period <= ts]


def ict_signal_for_symbol(
    symbol: str,
    bars_1min: pd.DataFrame,
    daily_bars: pd.DataFrame,
    qqq_1min: pd.DataFrame,
    session_date: date,
    equity: float,
    params: ICTParams = ICTParams(),
) -> Trade | None:
    """Scan one symbol on `session_date` and return at most one ICT entry.

    The first trigger encountered during a killzone wins. The function returns
    None if no setup forms.
    """
    if bars_1min is None or bars_1min.empty:
        return None
    if not is_trading_day(session_date):
        return None

    bars_1min = bars_1min.sort_index()
    bars_1h = _resample_to(bars_1min, "1h")
    bars_5m = _resample_to(bars_1min, "5min")

    daily_atr = _atr_14d(daily_bars, session_date, params.atr_window_daily)
    if not math.isfinite(daily_atr) or daily_atr <= 0:
        return None

    # Restrict 1-min bars to the session itself
    sess_start = datetime.combine(session_date, time(9, 30), tzinfo=NY_TZ)
    sess_end = datetime.combine(session_date, time(16, 0), tzinfo=NY_TZ)
    today_min = bars_1min[
        (bars_1min.index >= sess_start) & (bars_1min.index < sess_end)
    ]
    if today_min.empty:
        return None

    flat_t = force_flat_time(session_date)
    eod_dt = datetime.combine(session_date, flat_t, tzinfo=NY_TZ)

    # Walk 1-min bars. For each bar inside a killzone, evaluate full pipeline.
    for ts, bar in today_min.iterrows():
        t_local = ts.time()
        if not in_killzone(t_local):
            continue

        # 2. Bias gate (re-checked at every bar; cheap and adapts intraday).
        #    Mode-dispatched via params.bias_mode:
        #      "qqq"          -> QQQ uptrend required (original)
        #      "self"         -> the symbol itself uptrending (rel-strength)
        #      "qqq_or_self"  -> either qualifies
        #      "daily_self"   -> the symbol's daily 50-SMA > 200-SMA (most permissive)
        #      "none"         -> no bias filter
        if not bias_passes(
            mode=params.bias_mode,
            as_of=ts.to_pydatetime(),
            symbol_bars_1min=bars_1min,
            qqq_1min=qqq_1min,
            fast=params.bias_ema_fast,
            slow=params.bias_ema_slow,
            symbol_daily=daily_bars,
        ):
            continue

        # 3. HTF POI: price currently inside an unmitigated bullish 1H FVG.
        #    CAUSAL: only use 1H bars that have fully CLOSED by ts.
        htf_history = _closed_bars_at(bars_1h, ts, pd.Timedelta(hours=1))
        htf_fvgs = find_bullish_fvgs(
            htf_history, params.htf_lookback_bars, params.htf_fvg_min_pct
        )
        active_htf: FVG | None = None
        for fvg in htf_fvgs:
            after = htf_history[htf_history.index > fvg.start_ts]
            if not is_unmitigated(fvg, after):
                continue
            if price_inside_fvg(float(bar["low"]), fvg):
                active_htf = fvg
                break
        if active_htf is None:
            continue

        # 4. MTF: a bullish 5M FVG that started AFTER the HTF tap.
        #    CAUSAL: only use 5-min bars that have fully CLOSED by ts.
        mtf_history = _closed_bars_at(bars_5m, ts, pd.Timedelta(minutes=5))
        mtf_fvgs = find_bullish_fvgs(
            mtf_history, params.mtf_lookback_bars, params.mtf_fvg_min_pct
        )
        recent_mtf = [f for f in mtf_fvgs if f.start_ts >= active_htf.start_ts]
        if not recent_mtf:
            continue

        # 5. LTF entry: 1-min close > prior 1-min high (break of structure)
        prior_idx = today_min.index.get_loc(ts)
        if prior_idx == 0:
            continue
        prev_bar = today_min.iloc[prior_idx - 1]
        if not (bar["close"] > prev_bar["high"]):
            continue

        entry_price = float(bar["close"])

        # Risk: stop = lowest 1-min low in last N bars (incl. current)
        lookback_start = max(0, prior_idx - params.stop_lookback_bars + 1)
        lookback_window = today_min.iloc[lookback_start : prior_idx + 1]
        swing_low = float(lookback_window["low"].min())
        # Apply min-stop floor of 5% of daily ATR
        min_stop_dist = params.min_stop_pct_of_atr * daily_atr
        stop_price = min(swing_low, entry_price - min_stop_dist)
        stop_dist = entry_price - stop_price
        if stop_dist <= 0:
            continue

        risk_dollars = params.risk_per_trade * equity
        target_shares = math.floor(risk_dollars / stop_dist + 1e-6)
        if target_shares <= 0:
            continue
        # Leverage cap (single-trade)
        leverage_cap = params.max_leverage * equity
        notional = target_shares * entry_price
        if notional > leverage_cap:
            target_shares = math.floor(leverage_cap / entry_price + 1e-6)
            if target_shares <= 0:
                continue

        take_profit_price = (
            entry_price + params.take_profit_r * stop_dist
            if params.take_profit_r > 0
            else None
        )

        # Time exit = end of the killzone we're in, but never after EOD
        kz_end_t = killzone_end_for(t_local)
        kz_exit = datetime.combine(session_date, kz_end_t, tzinfo=NY_TZ)
        exit_dt = min(kz_exit, eod_dt)

        return Trade(
            symbol=symbol,
            side=Side.LONG,
            entry_time=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            entry_price=entry_price,
            stop_price=stop_price,
            shares=target_shares,
            exit_time=exit_dt,
            component="ict",
            take_profit_price=take_profit_price,
            meta={
                "atr_14d": daily_atr,
                "htf_fvg_top": active_htf.top,
                "htf_fvg_bottom": active_htf.bottom,
                "killzone_am": float(in_am_killzone(t_local)),
                "swing_low": swing_low,
            },
        )

    return None


def ict_signals(
    bars_1min_by_symbol: Mapping[str, pd.DataFrame],
    daily_bars_by_symbol: Mapping[str, pd.DataFrame],
    qqq_1min: pd.DataFrame,
    session_date: date,
    equity: float,
    params: ICTParams = ICTParams(),
) -> list[Trade]:
    """Run the ICT pipeline over the full universe for one session."""
    if not is_trading_day(session_date):
        return []
    out: list[Trade] = []
    for sym, bars in bars_1min_by_symbol.items():
        daily = daily_bars_by_symbol.get(sym, pd.DataFrame())
        t = ict_signal_for_symbol(
            sym,
            bars,
            daily,
            qqq_1min,
            session_date,
            equity,
            params,
        )
        if t is not None:
            out.append(t)
    return out
