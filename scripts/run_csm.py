"""CLI runner for cross-sectional momentum.

Usage:
    python -m scripts.run_csm --start 2023-01-01 --end 2024-12-31 \
        --symbols AAPL MSFT NVDA AMZN META GOOGL ... --top-n 5 --label csm_2023
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.backtest.metrics import (
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)
from src.data.ingest_alpaca import load_range
from src.strategies.cross_sectional_momentum import CSMParams, run_csm_backtest
from src.utils.logging import configure_logging, get_logger

log = get_logger("run_csm")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "csm.log")

    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--symbols", nargs="+", required=False)
    p.add_argument("--universe", default=None,
                   help="optional: 'sp100' for the curated SP100 list")
    p.add_argument("--use-daily-cache", action="store_true",
                   help="load from data/parquet_daily/{SYM}.parquet instead "
                        "of resampling minute bars")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--rebalance-days", type=int, default=21)
    p.add_argument("--lookback-days", type=int, default=252)
    p.add_argument("--skip-days", type=int, default=21)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--abs-momentum-threshold", type=float, default=0.0,
                   help="time-series filter: only hold names with r_12_1 > this")
    p.add_argument("--weight-scheme", choices=["equal", "inverse_vol"],
                   default="inverse_vol")
    p.add_argument("--vol-lookback-days", type=int, default=60)
    p.add_argument("--label", default="csm")
    args = p.parse_args(argv)

    if args.universe == "sp100":
        from config.sp100_universe import SP100_UNIVERSE
        symbols = list(SP100_UNIVERSE)
    elif args.symbols:
        symbols = list(args.symbols)
    else:
        log.error("csm.no_symbols")
        return 1

    start_d = datetime.fromisoformat(args.start).date()
    end_d = datetime.fromisoformat(args.end).date()
    log.info("csm.load", extra={"start": str(start_d), "end": str(end_d),
                                "n_symbols": len(symbols),
                                "use_daily_cache": args.use_daily_cache})

    daily_by_sym: dict[str, pd.DataFrame] = {}
    daily_dir = settings.data_dir / "parquet_daily"

    for sym in symbols:
        if args.use_daily_cache:
            p_path = daily_dir / f"{sym.upper()}.parquet"
            if not p_path.exists():
                log.warning("csm.daily_missing", extra={"symbol": sym})
                continue
            d = pd.read_parquet(p_path)
        else:
            m = load_range(sym, start_d, end_d, settings.data_dir)
            if m.empty:
                continue
            if m.index.tz is None:
                m = m.tz_localize("UTC")
            d = (
                m.resample("1D")
                .agg({"open": "first", "high": "max", "low": "min",
                      "close": "last", "volume": "sum"})
                .dropna()
            )
        if d.empty:
            continue
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        # Restrict to the request window
        d = d.loc[(d.index >= pd.Timestamp(start_d)) & (d.index <= pd.Timestamp(end_d))]
        if d.empty:
            continue
        daily_by_sym[sym] = d
        log.info("csm.symbol_loaded",
                 extra={"symbol": sym, "rows": int(len(d)),
                        "first": str(d.index[0].date()),
                        "last": str(d.index[-1].date())})

    log.info("csm.run", extra={"n_symbols": len(daily_by_sym)})
    result = run_csm_backtest(
        daily_by_sym,
        start=start_d,
        end=end_d,
        params=CSMParams(
            initial_equity=settings.initial_equity,
            top_n=args.top_n,
            rebalance_days=args.rebalance_days,
            lookback_days=args.lookback_days,
            skip_days=args.skip_days,
            slippage_bps=args.slippage_bps,
            abs_momentum_threshold=args.abs_momentum_threshold,
            weight_scheme=args.weight_scheme,
            vol_lookback_days=args.vol_lookback_days,
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
            "csm": float(sum(pnl_per_trade)),
            "orb": 0.0,
            "intraday_momentum": 0.0,
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
         "pnl_gross": t.pnl_gross, "commission": t.commission, "pnl_net": t.pnl_net}
        for t in result.trades
    ]).to_csv(out_dir / "trades.csv", index=False)
    (out_dir / "rebalances.json").write_text(json.dumps(result.rebalance_log, indent=2))

    log.info("csm.done", extra={"summary": str(out_dir / "summary.json"),
                                "n_trades": len(result.trades)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
