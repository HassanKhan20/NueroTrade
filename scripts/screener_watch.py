"""Long-running watcher that re-screens on a timer and emits a callouts feed.

Each cycle:
  1. Calls run_screener_once() (writes reports/screener_latest.json).
  2. Diffs current picks vs. the previous snapshot (kept in memory).
  3. Appends events (NEW / UPGRADED / DOWNGRADED / ACTION_CHANGED / DROPPED)
     to reports/callouts_feed.jsonl, newest line at the bottom.
  4. Sleeps until the next cycle.

The dashboard tails the feed file. Run this in its own terminal:

    python -m scripts.screener_watch \
        --interval 900 --universe halal --top-n 10 --equity 100000

By default we only run during US equity market hours (9:30 ET .. 16:00 ET,
Mon-Fri). Pass --always to ignore market hours.

Stop with Ctrl+C — events flush on each cycle so nothing is lost.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from config.settings import get_settings
from scripts.screener import UNIVERSE_CHOICES, run_screener_once
from src.signals.callout_diff import diff_picks
from src.utils.logging import configure_logging, get_logger

log = get_logger("screener_watch")

ET = ZoneInfo("America/New_York")
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)


def _is_market_hours(now_utc: datetime | None = None) -> bool:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(ET)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def _append_events(feed_path: Path, events: list[dict[str, Any]]) -> None:
    if not events:
        return
    feed_path.parent.mkdir(parents=True, exist_ok=True)
    with feed_path.open("a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, default=str) + "\n")


def _print_event(ev: dict[str, Any]) -> None:
    sym = ev.get("symbol", "?")
    grade = ev.get("grade", "?")
    kind = ev.get("event", "?")
    action = ev.get("action", "")
    entry = ev.get("entry_price")
    current = ev.get("current_price")
    ret = ev.get("return_pct")
    extra = ""
    if kind == "UPGRADED":
        extra = f" {ev.get('prev_grade','?')} -> {grade}"
    elif kind == "DOWNGRADED":
        extra = f" {ev.get('prev_grade','?')} -> {grade}"
    elif kind == "ACTION_CHANGED":
        extra = f" {ev.get('prev_action','?')[:20]} -> {action[:20]}"
    elif kind in ("TP1_HIT", "TP2_HIT", "STOP_HIT") and current is not None:
        ret_s = f" ({ret:+.1f}%)" if ret is not None else ""
        extra = f" @${current:.2f}{ret_s}"
    px = (f" entry ${entry:.2f}"
          if kind in ("NEW", "UPGRADED", "DOWNGRADED")
          and isinstance(entry, (int, float)) else "")
    print(f"[{ev.get('ts','')}] {kind:<14} {sym:<6} [{grade}]{extra}{px}")


def _load_feed_dedupe_keys(feed_path: Path, max_lines: int = 5000) -> set[tuple]:
    """Read recent feed entries to avoid emitting duplicate events on restart."""
    if not feed_path.exists():
        return set()
    keys: set[tuple] = set()
    try:
        with feed_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-max_lines:]:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((ev.get("symbol"), ev.get("event"),
                      ev.get("grade"), ev.get("ts")))
    except OSError:
        pass
    return keys


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    configure_logging(s.log_level, s.logs_dir / "screener_watch.log")

    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=900,
                   help="seconds between scans (default 900 = 15 min)")
    p.add_argument("--universe", default="all", choices=UNIVERSE_CHOICES,
                   help="default 'all' scans every halal+non-halal mega+mid; "
                        "use halal/midcap/sp100/biotech for narrower scans")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--equity", type=float, default=None)
    p.add_argument("--risk-pct", type=float, default=0.01)
    p.add_argument("--max-share-price", type=float, default=None,
                   help="drop picks whose entry price exceeds this dollar "
                        "amount per share — useful for small accounts")
    p.add_argument("--min-grade", default="C",
                   help="floor grade for feed events (F/D/C/B/A)")
    p.add_argument("--always", action="store_true",
                   help="scan continuously, ignoring US market hours")
    p.add_argument("--once", action="store_true",
                   help="run a single pass and exit")
    args = p.parse_args(argv)

    feed_path = s.reports_dir / "callouts_feed.jsonl"
    log.info("watch.start",
             extra={"interval": args.interval, "universe": args.universe,
                    "top_n": args.top_n, "feed": str(feed_path),
                    "always": args.always})

    print(f"NeuroTrade screener watcher")
    print(f"  universe : {args.universe} (top {args.top_n})")
    print(f"  interval : {args.interval}s")
    print(f"  market   : {'always' if args.always else 'US hours only (9:30-16:00 ET)'}")
    print(f"  feed     : {feed_path}")
    print(f"  Ctrl+C to stop")
    print()

    prev_picks: list[dict] | None = None
    seen_keys: set[tuple] = _load_feed_dedupe_keys(feed_path)
    stop = {"flag": False}

    def _on_signal(_sig, _frm):
        stop["flag"] = True
        print("\n[stopping after current cycle]")

    try:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
    except (ValueError, AttributeError):
        pass

    while not stop["flag"]:
        cycle_start = time.time()
        if not args.always and not _is_market_hours():
            log.info("watch.skip.outside_hours")
            print(f"[{datetime.now(ET).isoformat(timespec='seconds')}] "
                  f"outside US market hours — sleeping")
        else:
            try:
                output = run_screener_once(
                    universe=args.universe, top_n=args.top_n,
                    min_score=args.min_score,
                    equity=args.equity, risk_pct=args.risk_pct,
                    max_share_price=args.max_share_price,
                    settings=s, write_files=True,
                )
                curr_picks = output.get("picks", [])
                events = diff_picks(prev_picks, curr_picks,
                                    min_grade=args.min_grade)
                # Drop events whose key already exists (restart safety)
                fresh: list[dict[str, Any]] = []
                for ev in events:
                    k = (ev.get("symbol"), ev.get("event"),
                         ev.get("grade"), ev.get("ts"))
                    if k in seen_keys:
                        continue
                    seen_keys.add(k)
                    fresh.append(ev)
                _append_events(feed_path, fresh)
                for ev in fresh:
                    _print_event(ev)

                # ====== LIVE PRICE-LEVEL EVENTS ======
                # Walk the feed (now includes any NEW we just emitted) to
                # find still-open callouts. Pull live quotes from Alpaca.
                # Emit TP1_HIT / TP2_HIT / STOP_HIT for any newly-crossed
                # levels. This is what makes the feed actually MOVE
                # between daily closes.
                from src.signals.level_events import (
                    load_active_callouts, detect_level_events,
                    fetch_current_prices,
                )
                active = load_active_callouts(feed_path)
                price_events: list[dict] = []
                if active and s.alpaca_key and s.alpaca_secret:
                    prices = fetch_current_prices(
                        list(active.keys()), s.alpaca_key, s.alpaca_secret,
                    )
                    price_events = detect_level_events(active, prices)
                    # Dedupe against the feed (in case of restart)
                    fresh_price_events = []
                    for ev in price_events:
                        k = (ev.get("symbol"), ev.get("event"),
                             ev.get("entry_ts"))
                        if k in seen_keys:
                            continue
                        seen_keys.add(k)
                        fresh_price_events.append(ev)
                    _append_events(feed_path, fresh_price_events)
                    for ev in fresh_price_events:
                        _print_event(ev)
                    price_events = fresh_price_events

                if not fresh and not price_events:
                    n = len(curr_picks)
                    open_n = len(active) if 'active' in dir() else 0
                    print(f"[{datetime.now(ET).isoformat(timespec='seconds')}] "
                          f"no changes — {n} picks tracked, {open_n} open")
                prev_picks = curr_picks
                log.info("watch.cycle",
                         extra={"n_picks": len(curr_picks),
                                "n_events": len(fresh),
                                "n_price_events": len(price_events),
                                "n_active": len(active) if active else 0})
            except Exception as e:  # noqa: BLE001
                log.exception("watch.error", extra={"err": str(e)})
                print(f"[error] {e}")

        if args.once:
            break

        elapsed = time.time() - cycle_start
        sleep_for = max(5, args.interval - int(elapsed))
        # Sleep in 1s chunks so Ctrl+C is responsive
        for _ in range(sleep_for):
            if stop["flag"]:
                break
            time.sleep(1)

    log.info("watch.stop")
    print("watcher stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
