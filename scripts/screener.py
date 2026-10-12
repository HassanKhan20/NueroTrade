"""Daily ranked watchlist with GRADED CALLOUTS for the multi-factor swing strategy.

For each pick the screener emits:
  - Grade (A+ / A / B / C / D / F)
  - Composite z-score and sub-factor breakdown
  - Suggested entry, stop, two take-profit targets, ATR
  - Reward-to-risk ratio at TP1 and TP2
  - Suggested share count for a fixed risk-per-trade
  - Strengths / Weaknesses bullets
  - Action verb ("ENTER" / "WAIT" / "SKIP" etc.)

Writes to:
  reports/screener_latest.json   (consumed by the dashboard)
  reports/screener_log/{date}.json

The watcher (scripts/screener_watch.py) imports run_screener_once() to
re-screen on a timer and emit a callouts feed.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import Settings, get_settings
from src.signals.callout import build_callout, callout_to_dict
from src.strategies.multi_factor_swing import (
    MultiFactorParams,
    rank_universe,
)
from src.utils.logging import configure_logging, get_logger

log = get_logger("screener")


UNIVERSE_CHOICES = ["all", "halal", "midcap", "sp100", "biotech"]


def _load_universe(name: str) -> list[str]:
    if name == "all":
        # Master catalog: all halal mega + all halal mid + all non-halal
        # mega + all non-halal mid. Picks are tagged with cap_tier and
        # halal so the dashboard can filter.
        from config.universe_meta import all_symbols
        return list(all_symbols())
    if name == "halal":
        from config.halal_universe import HALAL_UNIVERSE
        return list(HALAL_UNIVERSE)
    if name == "midcap":
        from config.midcap_universe import MIDCAP_UNIVERSE
        return list(MIDCAP_UNIVERSE)
    if name == "sp100":
        from config.sp100_universe import SP100_UNIVERSE
        return list(SP100_UNIVERSE)
    if name == "biotech":
        from config.biotech_universe import BIOTECH_UNIVERSE
        return list(BIOTECH_UNIVERSE)
    raise ValueError(f"unknown universe: {name}")


def _load_daily_bars(symbols: list[str], data_dir: Path) -> dict[str, pd.DataFrame]:
    daily_dir = data_dir / "parquet_daily"
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = daily_dir / f"{sym.upper()}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        if d.empty:
            continue
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        out[sym] = d
    return out


def run_screener_once(
    *,
    universe: str = "halal",
    top_n: int = 10,
    min_score: float = 0.0,
    ref_date: date | None = None,
    equity: float | None = None,
    risk_pct: float = 0.01,
    max_share_price: float | None = None,
    settings: Settings | None = None,
    write_files: bool = True,
) -> dict[str, Any]:
    """Run one screener pass and return the output dict.

    Designed to be called both from the CLI (scripts.screener) and from
    the long-running watcher (scripts.screener_watch).
    """
    s = settings or get_settings()
    if ref_date is None:
        ref_date = datetime.now(timezone.utc).date()
    if equity is None:
        equity = s.initial_equity

    symbols = _load_universe(universe)
    daily_by_sym = _load_daily_bars(symbols, s.data_dir)

    log.info("screener.run",
             extra={"n_symbols": len(daily_by_sym), "as_of": str(ref_date),
                    "equity": equity, "universe": universe})

    params = MultiFactorParams()
    # Load earnings cache (built by scripts/build_earnings_cache.py).
    # When the cache is present, PEAD becomes a real signal — names that
    # recently reported a strong positive surprise get a boost during
    # the 60-day drift window.
    from src.signals.pead import load_earnings_events
    earnings_events = load_earnings_events()
    log.info("screener.earnings_loaded",
             extra={"n_events": len(earnings_events)})
    ranked = rank_universe(daily_by_sym, ref_date, earnings_events, params)
    qualified = [r for r in ranked if r[1] >= min_score]

    # When scanning everything, surface deeper into the qualified list so
    # the dashboard tabs (halal/non-halal × mega/mid) each have something
    # to show. The dashboard re-ranks within each filter view anyway.
    extended_n = max(top_n * 4, 40)
    candidates = qualified[:extended_n]

    callouts = []
    ref_ts = pd.Timestamp(ref_date)
    for sym, score, raws in candidates:
        bars = daily_by_sym.get(sym)
        if bars is None:
            continue
        c = build_callout(
            sym, bars, ref_ts, composite_z=score, raw_scores=raws,
            equity=equity, risk_per_trade_pct=risk_pct,
        )
        if c is None:
            continue
        if max_share_price is not None and c.entry_price > max_share_price:
            continue
        callouts.append(c)

    # Tag each pick with cap_tier + halal so the dashboard can filter
    # without needing to look anything up.
    from config.universe_meta import meta_for

    picks_out: list[dict[str, Any]] = []
    for i, c in enumerate(callouts):
        meta = meta_for(c.symbol) or {"cap": "mega", "halal": True}
        picks_out.append({
            "rank": i + 1,
            **callout_to_dict(c),
            "technical_score": c.raw_scores.get("tech", 0.0),
            "momentum_score": c.raw_scores.get("mom", 0.0),
            "quality_score": c.raw_scores.get("qual", 0.0),
            "pead_score": c.raw_scores.get("pead", 0.0),
            "cap_tier": meta["cap"],
            "halal": meta["halal"],
        })

    output = {
        "as_of": ref_date.isoformat(),
        "universe": universe,
        "n_universe": len(daily_by_sym),
        "n_qualified": len(qualified),
        "top_n": top_n,
        "min_score": min_score,
        "equity_assumed": equity,
        "risk_pct": risk_pct,
        "max_share_price": max_share_price,
        "picks": picks_out,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if write_files:
        s.reports_dir.mkdir(parents=True, exist_ok=True)
        (s.reports_dir / "screener_latest.json").write_text(
            json.dumps(output, indent=2, default=str)
        )
        log_dir = s.reports_dir / "screener_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{ref_date.isoformat()}.json").write_text(
            json.dumps(output, indent=2, default=str)
        )

    return output


def _print_terminal_summary(output: dict[str, Any], top_n: int) -> None:
    picks = output["picks"][:top_n]
    print()
    print(f"=== SCREENER (as of {output['as_of']}) ===")
    print(f"universe: {output['universe']} ({output['n_universe']} names)")
    print(f"qualified: {output['n_qualified']} above z={output['min_score']:.2f}")
    eq = output["equity_assumed"]
    print(f"equity: ${eq:,.0f} | risk/trade: ${eq*output['risk_pct']:,.2f}")
    print()
    for i, p in enumerate(picks, 1):
        print(f"--- #{i}  [{p['grade']:>2}]  {p['symbol']}  -- {p['action']}")
        rs = p.get("raw_scores", {})
        print(f"    composite z={p['composite_score']:+.2f}  "
              f"(tech {rs.get('tech',0):+.2f}  mom {rs.get('mom',0):+.2f}  "
              f"qual {rs.get('qual',0):+.2f}  pead {rs.get('pead',0):+.2f})")
        print(f"    entry ${p['entry_price']:.2f}  stop ${p['stop_price']:.2f}  "
              f"TP1 ${p['take_profit_1']:.2f}  TP2 ${p['take_profit_2']:.2f}  "
              f"ATR ${p['atr_14']:.2f}")
        print(f"    risk ${p['risk_per_share']:.2f}/sh  "
              f"R:R TP1 {p['reward_to_risk_tp1']:.2f}:1  "
              f"R:R TP2 {p['reward_to_risk_tp2']:.2f}:1  "
              f"size {p['suggested_shares']} sh (~${p['notional']:,.0f})")
        for st in p.get("strengths", []):
            print(f"    + {st}")
        for w in p.get("weaknesses", []):
            print(f"    - {w}")
        print()


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    configure_logging(s.log_level, s.logs_dir / "screener.log")

    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--universe", default="halal",
                   choices=UNIVERSE_CHOICES,
                   help="halal=mega-cap halal, midcap=halal mid-cap (cheaper "
                        "share prices for small accounts), sp100, biotech")
    p.add_argument("--date", default=None,
                   help="reference date (YYYY-MM-DD). Defaults to today UTC.")
    p.add_argument("--equity", type=float, default=None,
                   help="account equity for share-count sizing")
    p.add_argument("--risk-pct", type=float, default=0.01,
                   help="risk per trade as fraction of equity (default 1%)")
    p.add_argument("--max-share-price", type=float, default=None,
                   help="drop picks whose entry price exceeds this dollar "
                        "amount per share — useful for small accounts")
    args = p.parse_args(argv)

    ref = (datetime.fromisoformat(args.date).date() if args.date
           else datetime.now(timezone.utc).date())
    output = run_screener_once(
        universe=args.universe,
        top_n=args.top_n,
        min_score=args.min_score,
        ref_date=ref,
        equity=args.equity,
        risk_pct=args.risk_pct,
        max_share_price=args.max_share_price,
        settings=s,
        write_files=True,
    )
    _print_terminal_summary(output, args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
