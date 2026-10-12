"""Post-Earnings Announcement Drift signal — architecture-only stub.

PEAD requires a quarterly earnings calendar with:
  - announcement date
  - actual EPS
  - consensus estimate (or rolling history → SUE)

Without an earnings feed (Bloomberg / Refinitiv / Zacks / EarningsWhispers),
we can only build the framework and stub the data fetch. The strategy
literature (Bernard-Thomas 1989, Fink 2021 replication) reports:

  - Top SUE decile drifts up ~5-7% over 60 days post-announcement
  - Bottom SUE decile drifts down ~5-7%
  - Edge is largest in first 30 days, decays after 60

When you wire in an earnings feed, replace `_load_earnings_table()` with
a real fetch and the signal will start producing values.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EarningsEvent:
    symbol: str
    announcement_date: date
    actual_eps: float
    consensus_eps: float
    surprise_pct: float            # (actual - consensus) / abs(consensus)


def standardized_unexpected_earnings(events: list[EarningsEvent],
                                      symbol: str) -> float:
    """Compute SUE = surprise / std-dev-of-historical-surprises for symbol.
    Returns NaN if < 4 prior surprises."""
    sym_events = sorted(
        [e for e in events if e.symbol == symbol],
        key=lambda e: e.announcement_date,
    )
    if len(sym_events) < 5:
        return float("nan")
    most_recent = sym_events[-1]
    historical = [e.surprise_pct for e in sym_events[:-1]]
    sd = float(np.std(historical, ddof=1))
    if sd <= 0:
        return float("nan")
    return most_recent.surprise_pct / sd


def days_since_announcement(events: list[EarningsEvent], symbol: str,
                              ref_date: date) -> int | None:
    sym_events = [e for e in events if e.symbol == symbol
                                       and e.announcement_date <= ref_date]
    if not sym_events:
        return None
    return (ref_date - max(e.announcement_date for e in sym_events)).days


def pead_score(symbol: str, ref_date: date,
                earnings_events: list[EarningsEvent] | None = None) -> float:
    """Return a PEAD score for the symbol on ref_date.

    The score is positive when:
      - The most recent earnings announcement was a positive surprise
        (high SUE), AND
      - We are within the 60-day drift window post-announcement

    Returns 0.0 (neutral) when no earnings data is available — this is
    intentional: a missing feed should NOT default to "buy" or "skip,"
    it should default to "no opinion."
    """
    if not earnings_events:
        return 0.0
    days = days_since_announcement(earnings_events, symbol, ref_date)
    if days is None or days > 60 or days < 1:
        return 0.0  # outside drift window

    sue = standardized_unexpected_earnings(earnings_events, symbol)
    if not np.isfinite(sue):
        return 0.0

    # Decay with days: full weight days 1-30, linear decay days 30-60.
    if days <= 30:
        time_weight = 1.0
    else:
        time_weight = max(0.0, 1.0 - (days - 30) / 30.0)

    # Cap the SUE at +/- 3 to avoid outliers dominating.
    sue_clamped = float(np.clip(sue, -3.0, 3.0))
    return sue_clamped * time_weight


_EARNINGS_CACHE: list[EarningsEvent] | None = None


def _load_earnings_table() -> list[EarningsEvent]:
    """Load earnings events from config/earnings_cache.json (built by
    scripts/build_earnings_cache.py via yfinance).

    Cached in-memory after first read. Returns empty list if no cache
    file exists or it can't be parsed — PEAD score stays 0 (neutral).
    """
    global _EARNINGS_CACHE
    if _EARNINGS_CACHE is not None:
        return _EARNINGS_CACHE
    from pathlib import Path
    import json as _json

    path = Path(__file__).resolve().parents[2] / "config" / "earnings_cache.json"
    if not path.exists():
        _EARNINGS_CACHE = []
        return _EARNINGS_CACHE
    try:
        raw = _json.loads(path.read_text())
    except (OSError, _json.JSONDecodeError):
        _EARNINGS_CACHE = []
        return _EARNINGS_CACHE
    events: list[EarningsEvent] = []
    for sym, recs in raw.items():
        for r in recs or []:
            try:
                ad = datetime.fromisoformat(str(r["announcement_date"])).date()
                events.append(EarningsEvent(
                    symbol=sym,
                    announcement_date=ad,
                    actual_eps=float(r.get("actual_eps", 0.0)),
                    consensus_eps=float(r.get("consensus_eps", 0.0)),
                    surprise_pct=float(r.get("surprise_pct", 0.0)),
                ))
            except (ValueError, KeyError, TypeError):
                continue
    _EARNINGS_CACHE = events
    return events


def load_earnings_events() -> list[EarningsEvent]:
    """Public accessor — callers (screener, ranker) use this to fetch the
    cached earnings table. See ``_load_earnings_table``."""
    return _load_earnings_table()
