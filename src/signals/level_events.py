"""Live price-level event detection for active callouts.

The screener emits a NEW callout with entry / stop / TP1 / TP2 levels.
Until something crosses those levels, the only feed events are GRADE
changes — which rarely fire because composite-z is built from daily
closes. To make the feed feel alive, we also emit:

  - ``TP1_HIT``   when price crosses take_profit_1
  - ``TP2_HIT``   when price crosses take_profit_2
  - ``STOP_HIT``  when price crosses stop_price

The feed itself acts as our persistent state: we walk it on each cycle
to figure out which callouts are still "open" (NEW/UPGRADED without a
subsequent terminal event), then fetch current Alpaca quotes and emit
events for any newly-crossed levels.

This is pure, side-effect-free logic — the watcher script does the I/O.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_EVENTS = ("TP1_HIT", "TP2_HIT", "STOP_HIT")


@dataclass
class ActiveCallout:
    """An entry callout that hasn't yet been resolved."""
    symbol: str
    grade: str
    entry_ts: str
    entry_price: float
    stop_price: float
    take_profit_1: float
    take_profit_2: float
    cap_tier: str
    halal: bool
    fired: set[str] = field(default_factory=set)

    def is_closed(self) -> bool:
        """A callout is closed once TP2 or STOP hits — those are terminal.
        TP1 alone doesn't close the trade (runner stays on)."""
        return "TP2_HIT" in self.fired or "STOP_HIT" in self.fired


def _load_feed(feed_path: Path) -> list[dict[str, Any]]:
    if not feed_path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with feed_path.open("r", encoding="utf-8") as f:
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


def load_active_callouts(feed_path: Path) -> dict[str, ActiveCallout]:
    """Replay the feed in chronological order, return the symbols whose
    most recent entry hasn't yet hit TP2 or STOP. TP1 doesn't close.

    Logic per symbol:
      - NEW / UPGRADED → start fresh ActiveCallout (replaces any prior)
      - DROPPED        → remove the active callout
      - TP1_HIT        → mark fired (but keep open)
      - TP2_HIT / STOP_HIT → mark fired (closed; will be filtered out)
    """
    events = _load_feed(feed_path)
    callouts: dict[str, ActiveCallout] = {}
    for ev in events:
        sym = ev.get("symbol")
        if not sym:
            continue
        kind = ev.get("event")
        if kind in ("NEW", "UPGRADED"):
            try:
                callouts[sym] = ActiveCallout(
                    symbol=sym,
                    grade=str(ev.get("grade", "F")),
                    entry_ts=str(ev.get("ts", "")),
                    entry_price=float(ev.get("entry_price", 0)),
                    stop_price=float(ev.get("stop_price", 0)),
                    take_profit_1=float(ev.get("take_profit_1", 0)),
                    take_profit_2=float(ev.get("take_profit_2", 0)),
                    cap_tier=str(ev.get("cap_tier", "mega")),
                    halal=bool(ev.get("halal", True)),
                    fired=set(),
                )
            except (TypeError, ValueError):
                continue
        elif kind in TERMINAL_EVENTS:
            if sym in callouts:
                callouts[sym].fired.add(kind)
        elif kind == "DROPPED":
            callouts.pop(sym, None)
    return {s: c for s, c in callouts.items() if not c.is_closed()}


def _make_event(kind: str, c: ActiveCallout, price: float,
                 now_ts: str) -> dict[str, Any]:
    return {
        "ts": now_ts,
        "event": kind,
        "symbol": c.symbol,
        "grade": c.grade,
        "entry_price": c.entry_price,
        "stop_price": c.stop_price,
        "take_profit_1": c.take_profit_1,
        "take_profit_2": c.take_profit_2,
        "current_price": price,
        "return_pct": (price / c.entry_price - 1.0) * 100.0 if c.entry_price > 0 else 0.0,
        "cap_tier": c.cap_tier,
        "halal": c.halal,
        "entry_ts": c.entry_ts,
    }


def detect_level_events(
    active: dict[str, ActiveCallout],
    prices: dict[str, float],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """For each active callout with a known current price, emit any
    crossing events that haven't already fired for this entry.

    Priority: STOP first (conservative), then TP2 (which implies TP1),
    then TP1 on its own.

    Mutates each callout's ``fired`` set so subsequent calls in the same
    process don't double-emit.
    """
    now_ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    events: list[dict[str, Any]] = []
    for sym, c in active.items():
        price = prices.get(sym)
        if price is None:
            continue
        if c.is_closed():
            continue
        # Stop crossed below
        if "STOP_HIT" not in c.fired and price <= c.stop_price:
            events.append(_make_event("STOP_HIT", c, price, now_ts))
            c.fired.add("STOP_HIT")
            continue  # one terminal event per cycle
        # TP2 crossed (implies TP1 if not already)
        if "TP2_HIT" not in c.fired and price >= c.take_profit_2:
            if "TP1_HIT" not in c.fired:
                events.append(_make_event(
                    "TP1_HIT", c, c.take_profit_1, now_ts))
                c.fired.add("TP1_HIT")
            events.append(_make_event("TP2_HIT", c, price, now_ts))
            c.fired.add("TP2_HIT")
            continue
        # TP1 only
        if "TP1_HIT" not in c.fired and price >= c.take_profit_1:
            events.append(_make_event("TP1_HIT", c, price, now_ts))
            c.fired.add("TP1_HIT")
    return events


def fetch_current_prices(
    symbols: list[str], alpaca_key: str, alpaca_secret: str,
) -> dict[str, float]:
    """Pull latest trade prices via Alpaca. Returns {sym: price}. Missing
    symbols are silently omitted (delisted, rate-limited, etc.)."""
    if not symbols or not alpaca_key or not alpaca_secret:
        return {}
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
    except ImportError:
        return {}
    try:
        client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
        req = StockLatestTradeRequest(symbol_or_symbols=symbols)
        quotes = client.get_stock_latest_trade(req)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, float] = {}
    for sym, q in (quotes or {}).items():
        if q is None:
            continue
        try:
            out[sym] = float(q.price)
        except (AttributeError, TypeError, ValueError):
            continue
    return out
