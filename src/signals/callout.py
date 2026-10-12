"""Convert multi-factor scores into actionable graded entry callouts.

A callout is what you'd actually act on:
  - Grade (A+/A/B/C/D/F)
  - Suggested entry price
  - Stop-loss price (ATR-based or swing-low, whichever is tighter)
  - Two profit targets (TP1 = scale 50%, TP2 = runner)
  - Risk per share / reward-to-risk ratio
  - Suggested share count for a target risk-per-trade
  - Strengths / Weaknesses bullets
  - Suggested action ("enter now" / "wait for pullback" / "skip" etc.)

Grading is OBJECTIVE — derived from the composite z-score, the sub-factor
agreement, the trend filter, and the resulting risk:reward. No fudging.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.signals.technical import TechnicalSnapshot, technical_snapshot

Grade = Literal["A+", "A", "B", "C", "D", "F"]


@dataclass(frozen=True)
class Callout:
    """A complete entry recommendation for one symbol."""
    symbol: str
    grade: Grade
    composite_score: float            # z-scored composite from the universe
    raw_scores: dict[str, float]      # tech / mom / qual / pead

    # Trade levels
    entry_price: float
    stop_price: float
    take_profit_1: float              # scale 50% here
    take_profit_2: float              # runner — exit on trail or here
    atr_14: float

    # Derived
    risk_per_share: float
    reward_to_risk_tp1: float         # (TP1 - entry) / (entry - stop)
    reward_to_risk_tp2: float

    # Suggestions
    suggested_shares: int             # for a fixed-risk-per-trade size
    risk_dollars: float
    notional: float
    action: str                       # human-readable verb
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]

    def is_actionable(self) -> bool:
        return self.grade in ("A+", "A", "B", "C")


# ============================================================
# Grading rules — objective thresholds, not subjective judgement.
# ============================================================
def _grade_from_facts(
    composite_z: float,
    raws: dict[str, float],
    snap: TechnicalSnapshot,
    rr_tp1: float,
    risk_per_share: float = 0.0,
) -> Grade:
    """Objective grade based on:
      - composite_z (universe percentile proxy)
      - sub-factor agreement (how many of tech/mom/qual are positive)
      - trend filter (close > sma50 > sma200)
      - reward-to-risk ratio at TP1
      - non-fragile stop (risk_per_share >= 0.5 * ATR — learned from AEIS)

    Thresholds calibrated to bell-curve-ish z-scores: z=+2 ≈ 97th percentile,
    z=+1 ≈ 84th. We require BOTH signal strength AND a tradeable R:R AND
    a stop that's far enough from entry to survive normal volatility.

    Calibration history (LEARNED):
      - 60 callouts produced 1 A and 59 C (Grade B unreachable due to
        R:R 1.33 < 1.5). Default TP1 widened from 2.0x to 2.5x ATR, now
        R:R = 1.67 by default which lifts grades into B.
      - The one Grade A (AEIS) had a paper-thin swing-low stop. Grade A
        and A+ now require a non-fragile stop in addition to high z + R:R.
    """
    if not snap.is_in_uptrend():
        return "F"  # never enter against the trend
    if composite_z < 0:
        return "F"

    pos_subs = sum(1 for v in (raws.get("tech", 0), raws.get("mom", 0),
                                raws.get("qual", 0)) if v > 0)
    rsi = snap.rsi_14
    rsi_sane = 35 <= rsi <= 75 if np.isfinite(rsi) else True
    atr = float(snap.atr_14) if np.isfinite(snap.atr_14) else 0.0
    # A "real" stop = at least half an ATR away from entry. Anything
    # tighter means the swing-low fluke is masking a fragile setup.
    stop_robust = (atr > 0 and risk_per_share >= 0.5 * atr)

    if (composite_z >= 2.0 and pos_subs >= 3 and rr_tp1 >= 2.5
            and rsi_sane and snap.macd_hist > 0 and stop_robust):
        return "A+"
    if (composite_z >= 1.5 and pos_subs >= 3 and rr_tp1 >= 2.0
            and rsi_sane and stop_robust):
        return "A"
    if (composite_z >= 1.0 and pos_subs >= 2 and rr_tp1 >= 1.5):
        return "B"
    if composite_z >= 0.5 and rr_tp1 >= 1.0:
        return "C"
    if composite_z >= 0.0 and snap.is_in_uptrend():
        return "D"
    return "F"


# ============================================================
# Trade-level calculation: entry, stop, targets, sizing
# ============================================================
def _swing_low(closes: pd.Series, lows: pd.Series, n: int = 10) -> float:
    """Lowest low of the last n bars (excluding today). Returns NaN if
    insufficient history."""
    if len(lows) < n + 1:
        return float("nan")
    return float(lows.iloc[-(n + 1):-1].min())


def _build_levels(
    snap: TechnicalSnapshot,
    swing_low: float,
    atr_stop_mult: float = 1.5,
    atr_tp1_mult: float = 2.5,
    atr_tp2_mult: float = 4.0,
    min_stop_atr_frac: float = 0.5,
) -> tuple[float, float, float, float]:
    """Return (entry, stop, tp1, tp2) for a long entry today.

    Stop selection (LEARNED from AEIS failure):
      - Candidate A: entry - atr_stop_mult * ATR
      - Candidate B: 10-day swing low
      - Pick the tighter one — BUT reject candidate B if it sits less than
        ``min_stop_atr_frac * ATR`` below entry. A near-flat swing-low
        produces a fragile paper-thin stop that gives a misleadingly huge
        R:R, then gets blown out on the first wiggle (AEIS lesson).

    TP1 is at 2.5x ATR by default (LEARNED — previously 2.0x produced
    R:R 1.33, which was below every grading threshold above C).
    """
    entry = snap.last_close
    atr = snap.atr_14
    atr_stop = entry - atr_stop_mult * atr
    min_stop = entry - min_stop_atr_frac * atr  # nearest acceptable stop

    if np.isfinite(swing_low) and swing_low > 0 and swing_low < min_stop:
        # Swing low is far enough away to be a real stop — use the tighter
        # of (atr_stop, swing_low) so long as it stays below entry.
        stop = max(min(swing_low, entry * 0.999), atr_stop)
    else:
        # Either no swing low, or swing low is too close (fragile) — use
        # the pure ATR stop.
        stop = atr_stop

    if stop >= entry:
        # Pathological — fall back to ATR stop without floor
        stop = atr_stop
    tp1 = entry + atr_tp1_mult * atr
    tp2 = entry + atr_tp2_mult * atr
    return entry, stop, tp1, tp2


# ============================================================
# Factor diagnostics — what to TELL the user
# ============================================================
def _strengths_and_weaknesses(
    composite_z: float, raws: dict[str, float], snap: TechnicalSnapshot,
) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []

    if raws.get("mom", 0) > 0.5:
        strengths.append("12-month momentum strongly positive")
    elif raws.get("mom", 0) > 0:
        strengths.append("12-month momentum positive")
    elif raws.get("mom", 0) < 0:
        weaknesses.append("12-month momentum negative — counter-trend bet")

    if raws.get("tech", 0) > 1:
        strengths.append("Multiple technical signals aligned")
    if raws.get("qual", 0) >= 1:
        strengths.append("High-quality business (low debt + stable earnings)")
    elif raws.get("qual", 0) <= -1:
        weaknesses.append("Lower quality — cyclical or capital-intensive")

    if snap.is_in_uptrend():
        strengths.append("Above 50-SMA, 50-SMA above 200-SMA (uptrending)")
    else:
        weaknesses.append("Not in confirmed uptrend (close, 50-SMA, or 200-SMA misaligned)")

    if np.isfinite(snap.macd_hist) and snap.macd_hist > 0:
        strengths.append("MACD histogram positive (momentum building)")
    elif np.isfinite(snap.macd_hist) and snap.macd_hist < 0:
        weaknesses.append("MACD histogram negative (momentum fading)")

    if 50 <= snap.rsi_14 <= 70:
        strengths.append(f"RSI {snap.rsi_14:.0f} — strong but not overbought")
    elif snap.rsi_14 > 75:
        weaknesses.append(f"RSI {snap.rsi_14:.0f} — overbought, expect pullback")
    elif snap.rsi_14 < 40:
        weaknesses.append(f"RSI {snap.rsi_14:.0f} — oversold, may continue lower")

    if np.isfinite(snap.rel_volume):
        if snap.rel_volume >= 2:
            strengths.append(f"Volume {snap.rel_volume:.1f}x avg — strong participation")
        elif snap.rel_volume < 0.7:
            weaknesses.append("Volume below average — weak conviction")

    if snap.atr_14 / snap.last_close > 0.05:
        weaknesses.append(f"High ATR ({snap.atr_14/snap.last_close*100:.1f}% of price) — wider stops needed")

    return strengths, weaknesses


def _action_text(grade: Grade, snap: TechnicalSnapshot, rr_tp1: float) -> str:
    """Map (grade, technical state) -> action verb.

    Calibration history (LEARNED):
      - Original rule "RSI > 75 -> WAIT" called WAIT on MU (RSI 76, MACD
        strong) and it ran +55%. Same on GOOGL (RSI 78, MACD strong) +15%
        and CSCO +23%. RSI > 75 in a CONFIRMED uptrend with MACD positive
        is usually strong-trend continuation, NOT mean reversion.
      - DOCN at RSI 91 was correctly flagged WAIT (genuine extreme).

    New three-tier RSI logic:
      - RSI > 85           : ALWAYS WAIT (true extreme, even in strong trend)
      - RSI 75-85, MACD<0  : WAIT (overbought + momentum fading = real risk)
      - RSI 75-85, MACD>0  : ENTER scaled (strong trend; don't fight tape)
    """
    if grade == "F":
        return "SKIP — does not meet entry criteria"
    if grade == "D":
        return "WATCHLIST — marginal qualifier, prefer waiting for stronger setup"

    rsi = snap.rsi_14 if np.isfinite(snap.rsi_14) else 50.0
    macd_strong = np.isfinite(snap.macd_hist) and snap.macd_hist > 0

    # Tier 1: extreme RSI is always a wait, regardless of trend
    if rsi > 85:
        return (f"WAIT — RSI {rsi:.0f} EXTREME overbought, "
                f"scale in on pullback to 50-SMA")

    # Tier 2: overbought + momentum fading = real reversal risk
    if rsi > 75 and not macd_strong:
        return (f"WAIT — RSI {rsi:.0f} overbought and MACD fading, "
                f"wait for pullback or confirmation")

    # Tier 3: overbought BUT MACD confirming = strong trend, scale in
    # carefully (LEARNED from MU/GOOGL/CSCO that ran +20-55% from here)
    if rsi > 75 and macd_strong:
        return (f"ENTER half — RSI {rsi:.0f} stretched but MACD confirming "
                f"strong trend; tight stop, scale in")

    if rr_tp1 < 1.5:
        return "REDUCE SIZE — reward:risk below 1.5; consider half-size or skip"
    if grade in ("A+", "A"):
        return "ENTER — full size; scale 50% at TP1, trail to TP2"
    return "ENTER — half size; tight management"


# ============================================================
# Top-level builder
# ============================================================
def build_callout(
    symbol: str,
    bars: pd.DataFrame,
    ref_date: pd.Timestamp,
    composite_z: float,
    raw_scores: dict[str, float],
    *,
    equity: float = 25_000.0,
    risk_per_trade_pct: float = 0.01,
    atr_stop_mult: float = 1.5,
    atr_tp1_mult: float = 2.5,
    atr_tp2_mult: float = 4.0,
) -> Callout | None:
    """Generate a Callout for one symbol. Returns None if technical
    snapshot isn't computable (insufficient history)."""
    # Staleness guard: skip delisted/halted tickers whose freshest bar is
    # more than 7 days old (e.g. ANSS after the Synopsys acquisition) —
    # otherwise levels are computed off an ancient price.
    if bars is not None and not bars.empty:
        _last = bars.index[-1]
        if getattr(_last, "tzinfo", None) is not None:
            _last = _last.tz_localize(None)
        _ref = ref_date.tz_localize(None) if ref_date.tzinfo is not None else ref_date
        if (_ref - _last).days > 7:
            return None

    snap = technical_snapshot(bars, ref_date)
    if snap is None:
        return None

    # Need ATR + close to size anything.
    if not (np.isfinite(snap.atr_14) and snap.atr_14 > 0):
        return None

    df = bars.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    ts = ref_date.tz_localize(None) if ref_date.tzinfo is not None else ref_date
    prior = df.loc[df.index < ts]
    swing = _swing_low(prior["close"], prior["low"])

    entry, stop, tp1, tp2 = _build_levels(
        snap, swing,
        atr_stop_mult=atr_stop_mult,
        atr_tp1_mult=atr_tp1_mult,
        atr_tp2_mult=atr_tp2_mult,
    )
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return None
    rr_tp1 = (tp1 - entry) / risk_per_share
    rr_tp2 = (tp2 - entry) / risk_per_share

    risk_dollars = equity * risk_per_trade_pct
    suggested_shares = max(0, int(math.floor(risk_dollars / risk_per_share)))
    notional = suggested_shares * entry

    grade = _grade_from_facts(composite_z, raw_scores, snap, rr_tp1,
                               risk_per_share=risk_per_share)
    strengths, weaknesses = _strengths_and_weaknesses(composite_z, raw_scores, snap)
    action = _action_text(grade, snap, rr_tp1)

    return Callout(
        symbol=symbol,
        grade=grade,
        composite_score=composite_z,
        raw_scores=dict(raw_scores),
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


def callout_to_dict(c: Callout) -> dict:
    return {
        "symbol": c.symbol,
        "grade": c.grade,
        "composite_score": c.composite_score,
        "raw_scores": dict(c.raw_scores),
        "entry_price": c.entry_price,
        "stop_price": c.stop_price,
        "take_profit_1": c.take_profit_1,
        "take_profit_2": c.take_profit_2,
        "atr_14": c.atr_14,
        "risk_per_share": c.risk_per_share,
        "reward_to_risk_tp1": c.reward_to_risk_tp1,
        "reward_to_risk_tp2": c.reward_to_risk_tp2,
        "suggested_shares": c.suggested_shares,
        "risk_dollars": c.risk_dollars,
        "notional": c.notional,
        "action": c.action,
        "strengths": list(c.strengths),
        "weaknesses": list(c.weaknesses),
    }
