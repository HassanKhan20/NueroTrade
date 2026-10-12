"""Backtest runner for the multi-factor swing strategy. Reads daily bars
from the cache, runs the ensemble, writes a standard report."""
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
from src.strategies.multi_factor_swing import (
    MultiFactorParams,
    run_multi_factor_swing_backtest,
)
from src.utils.logging import configure_logging, get_logger

log = get_logger("run_swing_backtest")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "swing_backtest.log")

    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--symbols", nargs="+", required=False)
    p.add_argument("--universe", default="halal",
                   choices=["halal", "midcap", "sp100", "biotech"])
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--rebalance-days", type=int, default=10)
    p.add_argument("--soft-stop-pct", type=float, default=0.08)
    p.add_argument("--w-tech", type=float, default=0.40)
    p.add_argument("--w-mom", type=float, default=0.30)
    p.add_argument("--w-qual", type=float, default=0.20)
    p.add_argument("--w-pead", type=float, default=0.10)
    p.add_argument("--label", default="swing")
    args = p.parse_args(argv)

    if args.symbols:
        symbols = list(args.symbols)
    elif args.universe == "halal":
        from config.halal_universe import HALAL_UNIVERSE
        symbols = list(HALAL_UNIVERSE)
    elif args.universe == "midcap":
        from config.midcap_universe import MIDCAP_UNIVERSE
        symbols = list(MIDCAP_UNIVERSE)
    elif args.universe == "sp100":
        from config.sp100_universe import SP100_UNIVERSE
        symbols = list(SP100_UNIVERSE)
    elif args.universe == "biotech":
        from config.biotech_universe import BIOTECH_UNIVERSE
        symbols = list(BIOTECH_UNIVERSE)
    else:
        log.error("swing.no_symbols")
        return 1

    start_d = datetime.fromisoformat(args.start).date()
    end_d = datetime.fromisoformat(args.end).date()
    daily_dir = settings.data_dir / "parquet_daily"

    daily_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = daily_dir / f"{sym.upper()}.parquet"
        if not path.exists():
            log.warning("swing.missing_cache", extra={"symbol": sym})
            continue
        d = pd.read_parquet(path)
        if d.empty:
            continue
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        d = d.loc[(d.index >= pd.Timestamp(start_d) - pd.Timedelta(days=400))
                  & (d.index <= pd.Timestamp(end_d))]
        if d.empty:
            continue
        daily_by_sym[sym] = d

    log.info("swing.run", extra={"n_symbols": len(daily_by_sym),
                                  "universe": args.universe})
    result = run_multi_factor_swing_backtest(
        daily_by_sym, start=start_d, end=end_d, earnings_events=None,
        params=MultiFactorParams(
            initial_equity=settings.initial_equity,
            top_n=args.top_n,
            rebalance_days=args.rebalance_days,
            soft_stop_pct=args.soft_stop_pct,
            weight_technical=args.w_tech,
            weight_momentum=args.w_mom,
            weight_quality=args.w_qual,
            weight_pead=args.w_pead,
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
            "multi_factor_swing": float(sum(pnl_per_trade)),
            "csm": 0.0, "orb": 0.0, "intraday_momentum": 0.0,
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
        {
            "symbol": t.symbol, "entry_date": t.entry_date,
            "entry_price": t.entry_price, "exit_date": t.exit_date,
            "exit_price": t.exit_price, "shares": t.shares,
            "pnl_gross": t.pnl_gross, "commission": t.commission,
            "pnl_net": t.pnl_net, "composite_score": t.composite_score,
            "technical_score": t.technical_score,
            "momentum_score": t.momentum_score,
            "quality_score": t.quality_score,
            "pead_score": t.pead_score,
        }
        for t in result.trades
    ]).to_csv(out_dir / "trades.csv", index=False)
    (out_dir / "rebalances.json").write_text(
        json.dumps(result.rebalance_log, indent=2, default=str)
    )

    log.info("swing.done", extra={"summary": str(out_dir / "summary.json"),
                                   "n_trades": len(result.trades)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
