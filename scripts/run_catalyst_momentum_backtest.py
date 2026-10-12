"""Backtest-only runner for catalyst-momentum.

NOT a live runner — does not touch Alpaca order endpoints. Reads cached
daily bars from data/parquet_daily/ and writes a standard report under
reports/backtests/.

Usage:
    python -m scripts.run_catalyst_momentum_backtest \
        --start 2024-01-01 --end 2026-04-30 --label catalyst_2024
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.backtest.metrics import (
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)
from src.strategies.catalyst_momentum import (
    CatalystMomentumParams,
    run_catalyst_momentum_backtest,
)
from src.utils.logging import configure_logging, get_logger

log = get_logger("run_catalyst_momentum_bt")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "catalyst_bt.log")

    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--symbols", nargs="+", required=False)
    p.add_argument("--universe", default="biotech",
                   choices=["biotech", "sp100"])
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--hold-days", type=int, default=5)
    p.add_argument("--spike-threshold", type=float, default=3.0)
    p.add_argument("--min-momentum", type=float, default=0.0)
    p.add_argument("--slippage-bps", type=float, default=25.0)
    p.add_argument("--label", default="catalyst")
    args = p.parse_args(argv)

    if args.symbols:
        symbols = list(args.symbols)
    elif args.universe == "biotech":
        from config.biotech_universe import BIOTECH_UNIVERSE
        symbols = list(BIOTECH_UNIVERSE)
    elif args.universe == "sp100":
        from config.sp100_universe import SP100_UNIVERSE
        symbols = list(SP100_UNIVERSE)
    else:
        log.error("catalyst_bt.no_symbols")
        return 1

    start_d = datetime.fromisoformat(args.start).date()
    end_d = datetime.fromisoformat(args.end).date()
    daily_dir = settings.data_dir / "parquet_daily"

    daily_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = daily_dir / f"{sym.upper()}.parquet"
        if not path.exists():
            log.warning("catalyst_bt.missing", extra={"symbol": sym})
            continue
        d = pd.read_parquet(path)
        if d.empty:
            continue
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        d = d.loc[(d.index >= pd.Timestamp(start_d) - pd.Timedelta(days=60))
                  & (d.index <= pd.Timestamp(end_d))]
        if d.empty:
            continue
        daily_by_sym[sym] = d

    log.info("catalyst_bt.run", extra={"n_symbols": len(daily_by_sym)})
    result = run_catalyst_momentum_backtest(
        daily_by_sym, start=start_d, end=end_d,
        params=CatalystMomentumParams(
            initial_equity=settings.initial_equity,
            top_n=args.top_n,
            hold_days=args.hold_days,
            spike_threshold=args.spike_threshold,
            min_momentum=args.min_momentum,
            slippage_bps=args.slippage_bps,
        ),
    )

    pnl_per_trade = [t.pnl_net for t in result.trades]
    eq = result.equity_curve
    if len(eq):
        terminal = float(eq.iloc[-1])
        total_ret = terminal / float(eq.iloc[0]) - 1.0
    else:
        terminal, total_ret = 0.0, 0.0

    summary = {
        "label": args.label,
        "n_days": int(len(eq)),
        "n_trades": int(len(result.trades)),
        "n_rebalances": int(len(result.rebalance_log)),
        "terminal_equity": terminal,
        "total_return": total_ret,
        "sharpe": sharpe_ratio(result.daily_returns.to_numpy()),
        "max_drawdown": max_drawdown(eq.to_numpy()) if len(eq) else 0.0,
        "profit_factor": profit_factor(pnl_per_trade),
        "win_rate": win_rate(pnl_per_trade),
        "component_attribution": {
            "catalyst_momentum": float(sum(pnl_per_trade)),
            "orb": 0.0, "intraday_momentum": 0.0, "ict": 0.0,
        },
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_dir = settings.reports_dir / "backtests" / (
        f"{args.label}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    eq.to_csv(out_dir / "equity_curve.csv", header=["equity"])
    pd.DataFrame([
        {"symbol": t.symbol, "entry_date": t.entry_date, "entry_price": t.entry_price,
         "exit_date": t.exit_date, "exit_price": t.exit_price, "shares": t.shares,
         "pnl_gross": t.pnl_gross, "commission": t.commission, "pnl_net": t.pnl_net,
         "score": t.score, "vol_ratio": t.vol_ratio, "mom_5d": t.mom_5d}
        for t in result.trades
    ]).to_csv(out_dir / "trades.csv", index=False)
    (out_dir / "rebalances.json").write_text(
        json.dumps(result.rebalance_log, indent=2, default=str)
    )

    log.info("catalyst_bt.done",
             extra={"summary": str(out_dir / "summary.json"),
                    "n_trades": len(result.trades)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
