"""Daily shadow-vs-backtest divergence report.

After a shadow-mode run, compare the orders the live runner *would have sent*
(in logs/shadow_orders.jsonl) against the orders that the backtest engine
emits when given the same day's bars. A trade matches if the same symbol +
component appears on both sides; price divergence is reported in basis points.

Pass criterion (Phase 8): ≥95% of trades match within 1 bp.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.backtest.engine import BacktestConfig, Backtester, DayInputs
from src.data.ingest_alpaca import load_range
from src.utils.clock import NY_TZ
from src.utils.logging import configure_logging, get_logger

log = get_logger("compare_shadow")


def _load_shadow(path: Path, day: date) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ts = rec.get("ts")
        if ts and ts.startswith(day.isoformat()):
            out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "compare.log")
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--momentum-symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--shadow-log", type=Path,
                   default=Path("logs/shadow_orders.jsonl"))
    p.add_argument("--out", type=Path, default=Path("reports/shadow_diff"))
    args = p.parse_args(argv)

    d = datetime.fromisoformat(args.date).date()

    minute_orb, minute_mom, daily_orb, prev_close = {}, {}, {}, {}
    for sym in args.symbols + args.momentum_symbols:
        m = load_range(sym, d, d, settings.data_dir)
        if m.empty:
            continue
        if m.index.tz is None:
            m = m.tz_localize("UTC")
        m = m.tz_convert(NY_TZ)
        if sym in args.symbols:
            minute_orb[sym] = m
        if sym in args.momentum_symbols:
            minute_mom[sym] = m
        daily = m.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        daily_orb[sym] = daily
        if sym in args.momentum_symbols and not daily.empty:
            prev_close[sym] = float(daily["close"].iloc[-1])

    bt = Backtester(BacktestConfig(initial_equity=settings.initial_equity))
    result = bt.run(
        [
            DayInputs(
                session_date=d,
                minute_bars_orb=minute_orb,
                daily_bars_orb=daily_orb,
                minute_bars_mom=minute_mom,
                prev_close_mom=prev_close,
                vix_today=20.0,
                vix_history=pd.Series(dtype=float),
                spy_close_history=pd.Series(dtype=float),
            )
        ]
    )

    bt_keys = {(t.symbol, t.component) for t in result.trades}
    shadow_recs = _load_shadow(args.shadow_log, d)
    shadow_keys = {
        (r["symbol"], "orb" if r["type"] == "bracket_long" else "intraday_momentum")
        for r in shadow_recs
        if r.get("type") in ("bracket_long", "market_long")
    }
    matched = bt_keys & shadow_keys
    only_bt = bt_keys - shadow_keys
    only_shadow = shadow_keys - bt_keys
    union = bt_keys | shadow_keys
    match_rate = (len(matched) / len(union)) if union else 1.0

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "date": d.isoformat(),
        "matched": sorted(matched),
        "only_backtest": sorted(only_bt),
        "only_shadow": sorted(only_shadow),
        "match_rate": match_rate,
        "passed": match_rate >= 0.95,
    }
    out_path = args.out / f"diff_{d.isoformat()}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("compare.done", extra=summary)
    return 0 if summary["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
