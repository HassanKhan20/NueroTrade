"""Larry Williams volatility-breakout swing strategy — long-only adaptation.

Faithful to the rules in *Long-Term Secrets to Short-Term Trading*:

  - Range          = prior day High - prior day Low
  - Breakout long  = prior Close + K * Range   (K ~ 0.5, the "noise ratio")
  - Entry          = BUY-STOP at the trigger. You only get long if today's
                     price actually breaks the level — momentum must confirm.
                     (This is the key difference from a blind close-buy.)
  - Stop           = midpoint between prior Low and the entry  (Williams'
                     classic stop) with an ATR floor so it isn't paper-thin.
  - Targets        = multiples of the breakout Range (2x scale, 3x runner).
  - Williams %R(10) = his own oscillator, used as a momentum filter — avoid
                     buying when %R is pinned at the top (overbought
                     exhaustion) and prefer entries with room to run.

HONEST NOTE: the famous 11,000% (1987 World Cup) came from futures leverage
+ aggressive Kelly sizing, NOT the raw signal. This module sizes long-only
stock positions at a fixed fraction of equity. Expect a fraction of that
return and validate with CPCV before trusting it.

Long-only constraints (NueroTrade house rules): no shorts, requires the name
to be in a confirmed uptrend (close > 50-SMA > 200-SMA).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.signals.callout import Callout, Grade
from src.signals.technical import technical_snapshot

# Larry Williams defaults
DEFAULT_K = 0.5            # breakout "noise ratio" multiplier
PCT_R_PERIOD = 10          # Williams %R lookback (his classic)
RANGE_TP1_MULT = 2.0       # first target = entry + 2 * range
RANGE_TP2_MULT = 3.0       # runner = entry + 3 * range
# LEARNED from the parameter sweep (backtest_williams.py): a tight 0.5-ATR
# stop has NEGATIVE expectancy (19% win rate, PF 0.74) because breakouts
# normally retest before running. A 2.0-ATR stop flips it positive
# (51% win, +0.28%/trade, PF 1.17). So the stop floor is 2.0 ATR.
STOP_ATR_MULT = 2.0        # stop = entry - 2.0 * ATR (or wider via midpoint)


def _bars_fresh(bars: pd.DataFrame, ref_date: pd.Timestamp,
                max_stale_days: int = 7) -> bool:
    """True if the most recent bar is within max_stale_days of ref_date.
    Guards against delisted/halted tickers with ancient last bars."""
    if bars is None or bars.empty:
        return False
    idx = bars.index
    last = idx[-1]
    if getattr(last, "tzinfo", None) is not None:
        last = last.tz_localize(None)
    ref = ref_date.tz_localize(None) if ref_date.tzinfo is not None else ref_date
    return (ref - last).days <= max_stale_days


def williams_pct_r(bars: pd.DataFrame, period: int = PCT_R_PERIOD) -> float:
    """Williams %R for the most recent bar.

      %R = (HighestHigh - Close) / (HighestHigh - LowestLow) * -100

    Ranges -100 (most oversold) to 0 (most overbought). Returns NaN if
    insufficient history.
    """
    if len(bars) < period:
        return float("nan")
    window = bars.iloc[-period:]
    hh = float(window["high"].max())
    ll = float(window["low"].min())
    close = float(bars["close"].iloc[-1])
    rng = hh - ll
    if rng <= 0:
        return float("nan")
    return (hh - close) / rng * -100.0


def breakout_levels(
    bars: pd.DataFrame, ref_ts: pd.Timestamp, k: float = DEFAULT_K,
) -> tuple[float, float, float] | None:
    """Return (prior_close, prior_range, trigger) for the long breakout.

    Uses the bar PRIOR to ref_ts so there's no look-ahead: the trigger for
    today is computed entirely from yesterday's H/L/C.
    """
    df = bars
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    ts = ref_ts.tz_localize(None) if ref_ts.tzinfo is not None else ref_ts
    prior = df.loc[df.index < ts]
    if len(prior) < 2:
        return None
    last = prior.iloc[-1]
    prior_close = float(last["close"])
    prior_range = float(last["high"]) - float(last["low"])
    if prior_range <= 0:
        return None
    trigger = prior_close + k * prior_range
    return prior_close, prior_range, trigger


# ============================================================
# Grading — breakout strength + %R + trend
# ============================================================
VOL_CONFIRM_MULT = 1.5     # breakout bar volume must be >= 1.5x 20-day avg


def _grade_williams(
    in_uptrend: bool,
    triggered: bool,
    pct_r: float,
    range_to_atr: float,
    rr_tp1: float,
    rel_volume: float = 1.0,
) -> Grade:
    """Objective grade for a Williams breakout setup.

      - Must be in an uptrend (long-only) and have actually triggered.
      - VOLUME CONFIRMATION (LEARNED — the single biggest edge in the
        backtest: filtering for breakout-bar volume >= 1.5x average lifted
        per-trade profit factor from 1.17 to 2.08). High grades REQUIRE it.
      - Williams %R should show momentum with room: best around -55..-15.
      - A bigger prior range relative to ATR = a more decisive breakout.
      - R:R at TP1 must be tradeable.
    """
    if not in_uptrend or not triggered:
        return "F"
    pr_ok = np.isfinite(pct_r)
    strong_mom = pr_ok and (-55.0 <= pct_r <= -15.0)
    ok_mom = pr_ok and (-80.0 <= pct_r <= -10.0)
    big_range = range_to_atr >= 1.2
    ok_range = range_to_atr >= 0.8
    vol_confirmed = np.isfinite(rel_volume) and rel_volume >= VOL_CONFIRM_MULT
    vol_ok = np.isfinite(rel_volume) and rel_volume >= 1.2

    # A/B now REQUIRE volume confirmation — a breakout on weak volume is noise.
    if strong_mom and big_range and rr_tp1 >= 2.0 and vol_confirmed:
        return "A"
    if ok_mom and ok_range and rr_tp1 >= 1.5 and vol_ok:
        return "B"
    if ok_mom and rr_tp1 >= 1.0:
        return "C"
    if rr_tp1 >= 1.0:
        return "D"
    return "F"


def _action_williams(grade: Grade, triggered: bool, trigger: float,
                      last_close: float, pct_r: float) -> str:
    if grade == "F":
        return "SKIP — no confirmed breakout in uptrend"
    if not triggered:
        return (f"WAIT — set BUY-STOP at ${trigger:.2f}; only enter if price "
                f"breaks the volatility trigger")
    if np.isfinite(pct_r) and pct_r > -10:
        return (f"REDUCE — %R {pct_r:.0f} pinned/overbought; breakout "
                f"confirmed but late, half size or wait for retest")
    if grade in ("A", "B"):
        return ("ENTER — breakout confirmed; scale 50% at TP1, trail runner "
                "to TP2, time-stop after 10 bars")
    return "ENTER half — breakout confirmed but mixed; tight management"


def build_williams_callout(
    symbol: str,
    bars: pd.DataFrame,
    ref_date: pd.Timestamp,
    *,
    k: float = DEFAULT_K,
    equity: float = 25_000.0,
    risk_per_trade_pct: float = 0.01,
    range_tp1_mult: float = RANGE_TP1_MULT,
    range_tp2_mult: float = RANGE_TP2_MULT,
) -> Callout | None:
    """Build a Williams volatility-breakout callout for one symbol.

    Entry is the breakout trigger (a buy-stop). Stop is the midpoint of
    (prior low, entry) with an ATR floor. Targets are range-multiples.
    Returns None if the snapshot or breakout levels aren't computable.
    """
    # STALENESS GUARD: a delisted/halted ticker (e.g. ANSS after the
    # Synopsys acquisition) has a last bar weeks/months old. Its breakout
    # "trigger" would be computed off an ancient price with no relation to
    # the live quote — producing nonsense callouts. Skip if the freshest
    # bar is more than 7 days behind the scan date.
    if not _bars_fresh(bars, ref_date, max_stale_days=7):
        return None

    snap = technical_snapshot(bars, ref_date)
    if snap is None:
        return None
    if not (np.isfinite(snap.atr_14) and snap.atr_14 > 0):
        return None

    lvls = breakout_levels(bars, ref_date, k=k)
    if lvls is None:
        return None
    prior_close, prior_range, trigger = lvls

    entry = trigger
    # Williams' classic stop is the midpoint of (prior low, entry). The
    # parameter sweep showed that stop needs >= 2.0 ATR of breathing room
    # or breakout retests stop you out at a loss. So: use the WIDER of the
    # midpoint and a 2.0-ATR floor.
    df = bars
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    ts = ref_date.tz_localize(None) if ref_date.tzinfo is not None else ref_date
    prior = df.loc[df.index < ts]
    prior_low = float(prior.iloc[-1]["low"])
    mid_stop = (prior_low + entry) / 2.0
    atr_stop = entry - STOP_ATR_MULT * snap.atr_14
    stop = min(mid_stop, atr_stop)  # lower price = wider stop
    if stop >= entry:
        stop = atr_stop

    tp1 = entry + range_tp1_mult * prior_range
    tp2 = entry + range_tp2_mult * prior_range

    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return None
    rr_tp1 = (tp1 - entry) / risk_per_share
    rr_tp2 = (tp2 - entry) / risk_per_share

    risk_dollars = equity * risk_per_trade_pct
    suggested_shares = max(0, int(math.floor(risk_dollars / risk_per_share)))
    notional = suggested_shares * entry

    pct_r = williams_pct_r(bars)
    range_to_atr = prior_range / snap.atr_14 if snap.atr_14 > 0 else 0.0
    # "Triggered" = today's high already exceeded the buy-stop trigger.
    last_high = float(df.iloc[-1]["high"]) if len(df) else snap.last_close
    triggered = last_high >= trigger
    in_uptrend = snap.is_in_uptrend()

    rel_vol = snap.rel_volume if np.isfinite(snap.rel_volume) else 1.0
    grade = _grade_williams(in_uptrend, triggered, pct_r, range_to_atr, rr_tp1,
                            rel_volume=rel_vol)
    action = _action_williams(grade, triggered, trigger, snap.last_close, pct_r)

    strengths: list[str] = []
    weaknesses: list[str] = []
    if in_uptrend:
        strengths.append("In confirmed uptrend (close > 50-SMA > 200-SMA)")
    else:
        weaknesses.append("Not in uptrend — long-only filter would skip")
    if range_to_atr >= 1.2:
        strengths.append(f"Decisive range expansion ({range_to_atr:.1f}x ATR)")
    elif range_to_atr < 0.8:
        weaknesses.append(f"Weak range ({range_to_atr:.1f}x ATR) — soft breakout")
    if np.isfinite(pct_r):
        if -55 <= pct_r <= -15:
            strengths.append(f"Williams %R {pct_r:.0f} — momentum with room")
        elif pct_r > -10:
            weaknesses.append(f"Williams %R {pct_r:.0f} — overbought/exhausted")
        elif pct_r < -80:
            weaknesses.append(f"Williams %R {pct_r:.0f} — no momentum yet")
    if triggered:
        strengths.append("Breakout trigger already hit (long active)")
    else:
        weaknesses.append("Trigger not yet hit — pending buy-stop")
    if rel_vol >= VOL_CONFIRM_MULT:
        strengths.append(f"Volume {rel_vol:.1f}x avg — institutional confirmation")
    elif rel_vol < 1.0:
        weaknesses.append(f"Volume {rel_vol:.1f}x avg — weak/unconfirmed breakout")

    # composite rewards volume + decisive range + good momentum
    composite = (range_to_atr
                 + (1.0 if triggered else 0.0)
                 + min(rel_vol, 3.0)
                 + (0.5 if (np.isfinite(pct_r) and -55 <= pct_r <= -15) else 0.0))

    return Callout(
        symbol=symbol,
        grade=grade,
        composite_score=float(composite),
        raw_scores={
            "pct_r": float(pct_r) if np.isfinite(pct_r) else 0.0,
            "range_atr": float(range_to_atr),
            "trigger": float(trigger),
            "breakout_range": float(prior_range),
        },
        entry_price=entry,
        stop_price=stop,
        take_profit_1=tp1,
        take_profit_2=tp2,
        atr_14=snap.atr_14,
        risk_per_share=risk_per_share,
        reward_to_risk_tp1=rr_tp1,
        reward_to_risk_tp2=rr_tp2,
        suggested_shares=suggested_shares,
        risk_dollars=risk_dollars,
        notional=notional,
        action=action,
        strengths=tuple(strengths),
        weaknesses=tuple(weaknesses),
    )


def scan_williams(
    daily_by_sym: dict[str, pd.DataFrame],
    ref_date: pd.Timestamp,
    *,
    k: float = DEFAULT_K,
    equity: float = 25_000.0,
    risk_per_trade_pct: float = 0.01,
) -> list[Callout]:
    """Scan a universe for Williams breakout setups, return callouts sorted
    by grade then composite (triggered + decisive breakouts first)."""
    rank = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 0}
    out: list[Callout] = []
    for sym, bars in daily_by_sym.items():
        if bars is None or bars.empty:
            continue
        c = build_williams_callout(
            sym, bars, ref_date, k=k, equity=equity,
            risk_per_trade_pct=risk_per_trade_pct,
        )
        if c is not None and c.grade != "F":
            out.append(c)
    out.sort(key=lambda c: (rank.get(c.grade, 0), c.composite_score),
             reverse=True)
    return out
