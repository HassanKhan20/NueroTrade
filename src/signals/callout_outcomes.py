"""Self-grade past callouts by replaying actual price action against them.

For each NEW or UPGRADED event in reports/callouts_feed.jsonl, we know:
  - entry / stop / take_profit_1 / take_profit_2
  - timestamp the callout was emitted

After enough time has passed we can ask the bars cache: did price hit
the stop first, or hit TP1 first, or is the trade still open?

Outcome classes:
  - HIT_TP1   : price reached take_profit_1 BEFORE hitting stop
  - HIT_TP2   : price reached take_profit_2 BEFORE hitting stop
                (if TP2 hit, we credit it as HIT_TP2 even if TP1 also hit)
  - HIT_STOP  : price hit stop first (loss)
  - OPEN      : neither bound hit yet, still inside the window
  - EXPIRED   : window closed without hitting either bound (small win/loss)

Aggregation produces a CalloutAccuracy record the dashboard renders:
  - n_resolved, n_open, hit_rate, avg_return
  - breakdown by grade (A+/A/B/C/D)
  - breakdown by cap_tier (mega/mid)
  - breakdown by halal (True/False)

This is honest: callouts that haven't had time to play out are flagged
OPEN, not credited as wins. Callouts that hit stop are losses. We
compute returns net of nothing — gross — and let the user judge.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

Outcome = Literal["HIT_TP1", "HIT_TP2", "HIT_STOP", "OPEN", "EXPIRED", "SKIPPED"]


@dataclass
class CalloutOutcome:
    symbol: str
    grade: str
    cap_tier: str | None
    halal: bool | None
    event: str  # "NEW" | "UPGRADED"
    ts: str
    days_elapsed: float
    entry_price: float
    stop_price: float
    take_profit_1: float
    take_profit_2: float
    outcome: Outcome
    last_price: float | None
    return_pct: float | None  # since entry, on close


@dataclass
class AccuracyBucket:
    """Aggregate hit-rate / return stats for a slice of resolved callouts."""
    label: str
    n: int = 0
    n_resolved: int = 0
    n_open: int = 0
    n_tp1: int = 0
    n_tp2: int = 0
    n_stop: int = 0
    n_expired: int = 0
    avg_return_pct: float = 0.0
    hit_rate: float = 0.0  # (TP1+TP2) / resolved


@dataclass
class CalloutAccuracy:
    generated_at: str
    window_days: int
    n_total: int
    overall: AccuracyBucket
    by_grade: list[AccuracyBucket] = field(default_factory=list)
    by_cap: list[AccuracyBucket] = field(default_factory=list)
    by_halal: list[AccuracyBucket] = field(default_factory=list)
    recent: list[dict[str, Any]] = field(default_factory=list)


# ============================================================
# Loading the feed
# ============================================================
def load_feed(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


# ============================================================
# Resolving a single callout against bars
# ============================================================
def resolve_outcome(
    ev: dict[str, Any],
    bars: pd.DataFrame | None,
    *,
    horizon_days: int = 21,
    now: datetime | None = None,
) -> CalloutOutcome:
    """Replay the bars from the callout timestamp forward and classify.

    Walking rule: for each daily bar AFTER the callout date, check
    whether the day's HIGH crossed take_profit_1/_2 or the day's LOW
    crossed stop_price. The first hit wins. If both happened on the same
    day we conservatively assume the stop hit first (intra-day adverse
    move usually precedes the favourable extension).
    """
    now = now or datetime.now(timezone.utc)
    ts_str = str(ev.get("ts", ""))
    sym = str(ev.get("symbol", ""))
    grade = str(ev.get("grade", "F"))
    entry = float(ev.get("entry_price") or 0)
    stop = float(ev.get("stop_price") or 0)
    tp1 = float(ev.get("take_profit_1") or 0)
    tp2 = float(ev.get("take_profit_2") or 0)

    base = dict(
        symbol=sym, grade=grade,
        cap_tier=ev.get("cap_tier"),
        halal=ev.get("halal"),
        event=str(ev.get("event", "NEW")),
        ts=ts_str,
        entry_price=entry, stop_price=stop,
        take_profit_1=tp1, take_profit_2=tp2,
    )

    try:
        ev_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return CalloutOutcome(**base, days_elapsed=0.0, outcome="SKIPPED",
                              last_price=None, return_pct=None)

    if entry <= 0 or stop <= 0 or tp1 <= 0:
        return CalloutOutcome(**base, days_elapsed=0.0, outcome="SKIPPED",
                              last_price=None, return_pct=None)

    elapsed_days = (now - ev_dt).total_seconds() / 86400.0

    if bars is None or bars.empty:
        return CalloutOutcome(**base, days_elapsed=elapsed_days,
                              outcome="OPEN", last_price=None, return_pct=None)

    df = bars.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    cutoff = pd.Timestamp(ev_dt.replace(tzinfo=None))
    forward = df.loc[df.index > cutoff]
    if forward.empty:
        return CalloutOutcome(**base, days_elapsed=elapsed_days,
                              outcome="OPEN", last_price=None, return_pct=None)

    # Walk daily bars, oldest to newest.
    bars_in_window = forward.iloc[: max(1, horizon_days)]
    last_close = float(bars_in_window["close"].iloc[-1])
    return_pct = (last_close / entry - 1.0) * 100.0

    for _, row in bars_in_window.iterrows():
        hi = float(row["high"])
        lo = float(row["low"])
        # Conservative: same-day stop+TP both touched -> assume stop first
        if lo <= stop and hi >= tp1:
            return CalloutOutcome(**base, days_elapsed=elapsed_days,
                                  outcome="HIT_STOP", last_price=last_close,
                                  return_pct=return_pct)
        if lo <= stop:
            return CalloutOutcome(**base, days_elapsed=elapsed_days,
                                  outcome="HIT_STOP", last_price=last_close,
                                  return_pct=return_pct)
        if hi >= tp2:
            return CalloutOutcome(**base, days_elapsed=elapsed_days,
                                  outcome="HIT_TP2", last_price=last_close,
                                  return_pct=return_pct)
        if hi >= tp1:
            return CalloutOutcome(**base, days_elapsed=elapsed_days,
                                  outcome="HIT_TP1", last_price=last_close,
                                  return_pct=return_pct)

    # Window expired without touching anything
    if len(forward) >= horizon_days:
        return CalloutOutcome(**base, days_elapsed=elapsed_days,
                              outcome="EXPIRED", last_price=last_close,
                              return_pct=return_pct)
    return CalloutOutcome(**base, days_elapsed=elapsed_days,
                          outcome="OPEN", last_price=last_close,
                          return_pct=return_pct)


# ============================================================
# Aggregation
# ============================================================
def _bucket(label: str, outcomes: list[CalloutOutcome]) -> AccuracyBucket:
    b = AccuracyBucket(label=label)
    rets: list[float] = []
    for o in outcomes:
        b.n += 1
        if o.outcome in ("HIT_TP1", "HIT_TP2", "HIT_STOP", "EXPIRED"):
            b.n_resolved += 1
            if o.return_pct is not None:
                rets.append(o.return_pct)
        elif o.outcome == "OPEN":
            b.n_open += 1
        if o.outcome == "HIT_TP1": b.n_tp1 += 1
        elif o.outcome == "HIT_TP2": b.n_tp2 += 1
        elif o.outcome == "HIT_STOP": b.n_stop += 1
        elif o.outcome == "EXPIRED": b.n_expired += 1
    if rets:
        b.avg_return_pct = sum(rets) / len(rets)
    if b.n_resolved > 0:
        b.hit_rate = (b.n_tp1 + b.n_tp2) / b.n_resolved
    return b


def aggregate(
    outcomes: list[CalloutOutcome],
    *,
    window_days: int = 90,
) -> CalloutAccuracy:
    overall = _bucket("overall", outcomes)
    by_grade = [
        _bucket(g, [o for o in outcomes if o.grade == g])
        for g in ("A+", "A", "B", "C", "D")
    ]
    by_cap = [
        _bucket("mega", [o for o in outcomes if o.cap_tier == "mega"]),
        _bucket("mid",  [o for o in outcomes if o.cap_tier == "mid"]),
    ]
    by_halal = [
        _bucket("halal",     [o for o in outcomes if o.halal is True]),
        _bucket("non-halal", [o for o in outcomes if o.halal is False]),
    ]
    # Recent table — newest first, last 30 entries
    recent_sorted = sorted(outcomes, key=lambda o: o.ts, reverse=True)[:30]
    recent = [
        {
            "symbol": o.symbol, "grade": o.grade, "ts": o.ts,
            "days_elapsed": round(o.days_elapsed, 1),
            "outcome": o.outcome,
            "entry_price": o.entry_price, "last_price": o.last_price,
            "return_pct": (round(o.return_pct, 2) if o.return_pct is not None else None),
            "cap_tier": o.cap_tier, "halal": o.halal,
        }
        for o in recent_sorted
    ]
    return CalloutAccuracy(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        window_days=window_days,
        n_total=len(outcomes),
        overall=overall,
        by_grade=[b for b in by_grade if b.n > 0],
        by_cap=[b for b in by_cap if b.n > 0],
        by_halal=[b for b in by_halal if b.n > 0],
        recent=recent,
    )


# ============================================================
# Top-level entry — reads feed + bars cache, returns accuracy
# ============================================================
def compute_accuracy(
    feed_path: Path,
    bars_dir: Path,
    *,
    window_days: int = 90,
    horizon_days: int = 21,
    now: datetime | None = None,
) -> CalloutAccuracy:
    """Read the feed, resolve each NEW/UPGRADED event against the bars
    cache, and aggregate into a CalloutAccuracy record."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    events = load_feed(feed_path)
    relevant = [
        e for e in events
        if e.get("event") in ("NEW", "UPGRADED")
        and e.get("symbol") and e.get("entry_price")
    ]

    # Filter by window
    in_window: list[dict[str, Any]] = []
    for e in relevant:
        try:
            ts = datetime.fromisoformat(str(e.get("ts", "")).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if ts >= cutoff:
            in_window.append(e)

    # Dedupe by (symbol, ts) so a NEW + UPGRADED for the same callout
    # don't double-count.
    seen: set[tuple] = set()
    unique: list[dict[str, Any]] = []
    for e in sorted(in_window, key=lambda x: x.get("ts", "")):
        key = (e.get("symbol"), e.get("ts"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)

    # Lazy-load bars per symbol
    bars_cache: dict[str, pd.DataFrame] = {}

    def _bars_for(sym: str) -> pd.DataFrame | None:
        if sym in bars_cache:
            return bars_cache[sym]
        p = bars_dir / f"{sym.upper()}.parquet"
        if not p.exists():
            bars_cache[sym] = pd.DataFrame()
            return None
        try:
            d = pd.read_parquet(p)
            bars_cache[sym] = d
            return d
        except Exception:  # noqa: BLE001
            bars_cache[sym] = pd.DataFrame()
            return None

    outcomes: list[CalloutOutcome] = []
    for ev in unique:
        sym = str(ev.get("symbol", "")).upper()
        bars = _bars_for(sym)
        outcomes.append(resolve_outcome(
            ev, bars, horizon_days=horizon_days, now=now,
        ))

    return aggregate(outcomes, window_days=window_days)
