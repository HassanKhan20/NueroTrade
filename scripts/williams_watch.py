"""Long-running watcher for the Larry Williams volatility-breakout strategy.

Mirrors scripts/screener_watch.py but for Williams callouts, with WIN-STREAK
(anti-martingale) position sizing: after each winning callout the per-trade
risk grows, resetting to base after a loss. Capped at 3% so a hot streak
can't blow the account.

Each cycle:
  1. Read current win streak from the Williams feed -> risk_pct for sizing.
  2. Scan the universe for volume-confirmed breakouts (scan_williams).
  3. Diff vs the previous scan -> NEW / UPGRADED / DOWNGRADED / DROPPED.
  4. Price-level events (TP1_HIT / TP2_HIT / STOP_HIT) via live Alpaca quotes.
  5. Append everything to reports/williams_feed.jsonl.

  python -m scripts.williams_watch --interval 300 --universe all --equity 100000 --always

Writes reports/williams_latest.json each cycle (dashboard-readable).
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

import pandas as pd

from config.settings import get_settings
from scripts.screener import UNIVERSE_CHOICES, _load_daily_bars, _load_universe
from src.signals.callout import callout_to_dict
from src.signals.callout_diff import diff_picks
from src.signals.level_events import (
    detect_level_events,
    fetch_current_prices,
    load_active_callouts,
)
from src.signals.streak_sizing import current_win_streak, streak_risk_pct
from src.signals.williams_breakout import scan_williams
from src.utils.logging import configure_logging, get_logger

log = get_logger("williams_watch")

ET = ZoneInfo("America/New_York")
MARKET_OPEN, MARKET_CLOSE = dt_time(9, 30), dt_time(16, 0)


def _is_market_hours(now_utc: datetime | None = None) -> bool:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(ET)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def _append(feed_path: Path, events: list[dict[str, Any]]) -> None:
    if not events:
        return
    feed_path.parent.mkdir(parents=True, exist_ok=True)
    with feed_path.open("a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, default=str) + "\n")


def _print_event(ev: dict[str, Any]) -> None:
    sym, grade, kind = ev.get("symbol", "?"), ev.get("grade", "?"), ev.get("event", "?")
    cur, ret = ev.get("current_price"), ev.get("return_pct")
    extra = ""
    if kind in ("TP1_HIT", "TP2_HIT", "STOP_HIT") and cur is not None:
        rs = f" ({ret:+.1f}%)" if ret is not None else ""
        extra = f" @${cur:.2f}{rs}"
    print(f"[{ev.get('ts','')}] {kind:<12} {sym:<6} [{grade}]{extra}")


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    configure_logging(s.log_level, s.logs_dir / "williams_watch.log")

    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--universe", default="all", choices=UNIVERSE_CHOICES)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--equity", type=float, default=None)
    p.add_argument("--base-risk-pct", type=float, default=0.01)
    p.add_argument("--max-share-price", type=float, default=None)
    p.add_argument("--always", action="store_true")
    p.add_argument("--once", action="store_true")
    args = p.parse_args(argv)

    equity = args.equity if args.equity is not None else s.initial_equity
    feed_path = s.reports_dir / "williams_feed.jsonl"

    print("NeuroTrade — Larry Williams volatility-breakout watcher")
    print(f"  universe : {args.universe} (top {args.top_n})")
    print(f"  interval : {args.interval}s | base risk {args.base_risk_pct*100:.1f}% (win-streak scaled, cap 3%)")
    print(f"  feed     : {feed_path}")
    print()

    prev_picks: list[dict] | None = None
    seen: set[tuple] = set()
    # seed dedupe from existing feed
    if feed_path.exists():
        with feed_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    seen.add((e.get("symbol"), e.get("event"),
                              e.get("grade"), e.get("ts")))
                except (json.JSONDecodeError, AttributeError):
                    continue

    stop = {"flag": False}
    def _sig(_s, _f): stop["flag"] = True; print("\n[stopping]")
    try:
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
    except (ValueError, AttributeError):
        pass

    while not stop["flag"]:
        cycle_start = time.time()
        if not args.always and not _is_market_hours():
            print(f"[{datetime.now(ET).isoformat(timespec='seconds')}] outside market hours")
        else:
            try:
                # 1) win-streak -> risk pct
                streak = current_win_streak(feed_path)
                risk_pct = streak_risk_pct(streak, base_pct=args.base_risk_pct)

                # 2) scan
                daily = _load_daily_bars(_load_universe(args.universe), s.data_dir)
                ref = pd.Timestamp(datetime.now(timezone.utc).date())
                callouts = scan_williams(daily, ref, equity=equity,
                                         risk_per_trade_pct=risk_pct)
                if args.max_share_price is not None:
                    callouts = [c for c in callouts
                                if c.entry_price <= args.max_share_price]

                # Live prices for every pick so the callout shows current
                # market price next to the buy-stop trigger + a status.
                live_prices: dict[str, float] = {}
                if callouts and s.alpaca_key and s.alpaca_secret:
                    live_prices = fetch_current_prices(
                        [c.symbol for c in callouts], s.alpaca_key, s.alpaca_secret)

                from config.universe_meta import meta_for
                curr_picks = []
                for i, c in enumerate(callouts):
                    m = meta_for(c.symbol) or {"cap": "mega", "halal": True}
                    live = live_prices.get(c.symbol)
                    # Status of the breakout trigger vs live price:
                    #   PENDING  — live below trigger (waiting to break out)
                    #   ACTIVE   — live just above trigger, inside entry zone
                    #   EXTENDED — live already past TP1 (move happened, chasing)
                    if live is None:
                        status = "NO_QUOTE"
                    elif live < c.entry_price:
                        status = "PENDING"
                    elif live <= c.take_profit_1:
                        status = "ACTIVE"
                    else:
                        status = "EXTENDED"
                    curr_picks.append({
                        "rank": i + 1, **callout_to_dict(c),
                        "pct_r": c.raw_scores.get("pct_r", 0.0),
                        "range_atr": c.raw_scores.get("range_atr", 0.0),
                        "cap_tier": m["cap"], "halal": m["halal"],
                        "live_price": live,
                        "status": status,
                    })

                # 3) diff -> signal events
                events = diff_picks(prev_picks, curr_picks, min_grade="C")
                fresh = []
                for ev in events:
                    k = (ev.get("symbol"), ev.get("event"),
                         ev.get("grade"), ev.get("ts"))
                    if k in seen:
                        continue
                    seen.add(k)
                    fresh.append(ev)
                _append(feed_path, fresh)
                for ev in fresh:
                    _print_event(ev)

                # 4) price-level events
                active = load_active_callouts(feed_path)
                price_events = []
                if active and s.alpaca_key and s.alpaca_secret:
                    prices = fetch_current_prices(list(active.keys()),
                                                  s.alpaca_key, s.alpaca_secret)
                    pe = detect_level_events(active, prices)
                    for ev in pe:
                        k = (ev.get("symbol"), ev.get("event"), ev.get("entry_ts"))
                        if k in seen:
                            continue
                        seen.add(k)
                        price_events.append(ev)
                    _append(feed_path, price_events)
                    for ev in price_events:
                        _print_event(ev)

                # 5) write latest snapshot
                output = {
                    "as_of": ref.date().isoformat(),
                    "strategy": "williams_volatility_breakout",
                    "universe": args.universe,
                    "n_setups": len(callouts),
                    "win_streak": streak,
                    "risk_pct": risk_pct,
                    "equity_assumed": equity,
                    "picks": curr_picks,
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                (s.reports_dir / "williams_latest.json").write_text(
                    json.dumps(output, indent=2, default=str))

                prev_picks = curr_picks
                if not fresh and not price_events:
                    print(f"[{datetime.now(ET).isoformat(timespec='seconds')}] "
                          f"no changes — {len(curr_picks)} setups, streak {streak}, "
                          f"risk {risk_pct*100:.1f}%")
                log.info("williams_watch.cycle",
                         extra={"n_setups": len(callouts), "n_events": len(fresh),
                                "n_price_events": len(price_events),
                                "win_streak": streak, "risk_pct": risk_pct})
            except Exception as e:  # noqa: BLE001
                log.exception("williams_watch.error", extra={"err": str(e)})
                print(f"[error] {e}")

        if args.once:
            break
        sleep_for = max(5, args.interval - int(time.time() - cycle_start))
        for _ in range(sleep_for):
            if stop["flag"]:
                break
            time.sleep(1)

    print("williams watcher stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
