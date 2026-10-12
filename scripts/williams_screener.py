"""Larry Williams volatility-breakout swing screener.

Scans a universe for long-only breakout setups (trigger = prior close +
0.5 x prior range, in a confirmed uptrend, %R momentum filter), grades
them, and writes the same callout JSON the dashboard already renders.

  python -m scripts.williams_screener --universe all --top-n 10 --equity 25000

Output: reports/williams_latest.json (+ reports/williams_log/{date}.json)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from config.settings import get_settings
from scripts.screener import UNIVERSE_CHOICES, _load_daily_bars, _load_universe
from src.signals.callout import callout_to_dict
from src.signals.williams_breakout import scan_williams
from src.utils.logging import configure_logging, get_logger

log = get_logger("williams_screener")


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    configure_logging(s.log_level, s.logs_dir / "williams_screener.log")

    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="all", choices=UNIVERSE_CHOICES)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--k", type=float, default=0.5,
                   help="breakout noise-ratio multiplier (Williams ~0.5)")
    p.add_argument("--equity", type=float, default=None)
    p.add_argument("--risk-pct", type=float, default=0.01)
    p.add_argument("--date", default=None)
    p.add_argument("--max-share-price", type=float, default=None)
    args = p.parse_args(argv)

    ref = (datetime.fromisoformat(args.date).date() if args.date
           else datetime.now(timezone.utc).date())
    equity = args.equity if args.equity is not None else s.initial_equity

    symbols = _load_universe(args.universe)
    daily = _load_daily_bars(symbols, s.data_dir)
    log.info("williams.run", extra={"n_symbols": len(daily),
                                     "as_of": str(ref), "k": args.k})

    callouts = scan_williams(daily, pd.Timestamp(ref), k=args.k,
                             equity=equity, risk_per_trade_pct=args.risk_pct)
    if args.max_share_price is not None:
        callouts = [c for c in callouts if c.entry_price <= args.max_share_price]

    from config.universe_meta import meta_for
    picks = []
    for i, c in enumerate(callouts):
        m = meta_for(c.symbol) or {"cap": "mega", "halal": True}
        picks.append({
            "rank": i + 1, **callout_to_dict(c),
            "pct_r": c.raw_scores.get("pct_r", 0.0),
            "range_atr": c.raw_scores.get("range_atr", 0.0),
            "trigger": c.raw_scores.get("trigger", 0.0),
            "cap_tier": m["cap"], "halal": m["halal"],
        })

    output = {
        "as_of": ref.isoformat(),
        "strategy": "williams_volatility_breakout",
        "universe": args.universe,
        "n_universe": len(daily),
        "n_setups": len(callouts),
        "k": args.k,
        "equity_assumed": equity,
        "picks": picks,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    (s.reports_dir / "williams_latest.json").write_text(
        json.dumps(output, indent=2, default=str))
    log_dir = s.reports_dir / "williams_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{ref.isoformat()}.json").write_text(
        json.dumps(output, indent=2, default=str))

    print()
    print(f"=== WILLIAMS VOLATILITY BREAKOUT (as of {ref}) ===")
    print(f"universe: {args.universe} ({len(daily)} names) | "
          f"setups: {len(callouts)} | k={args.k}")
    print()
    for i, c in enumerate(callouts[: args.top_n], 1):
        rs = c.raw_scores
        print(f"--- #{i}  [{c.grade:>2}]  {c.symbol}  -- {c.action}")
        print(f"    %R {rs.get('pct_r',0):.0f}  range/ATR {rs.get('range_atr',0):.2f}  "
              f"trigger ${rs.get('trigger',0):.2f}")
        print(f"    entry ${c.entry_price:.2f}  stop ${c.stop_price:.2f}  "
              f"TP1 ${c.take_profit_1:.2f}  TP2 ${c.take_profit_2:.2f}  "
              f"R:R {c.reward_to_risk_tp1:.2f}")
        print(f"    size {c.suggested_shares} sh (~${c.notional:,.0f})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
