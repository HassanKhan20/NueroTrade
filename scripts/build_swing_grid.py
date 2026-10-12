"""CPCV parameter-grid runner for the multi-factor swing strategy.

Sweeps a small, principled grid:
    top_n            in {3, 5, 8}
    rebalance_days   in {5, 10, 21}
    soft_stop_pct    in {0.05, 0.08, 0.0}    # 0 = disabled

That's 27 variants — at the upper edge of what PBO can handle without
inflated multiple-testing penalty.

Output: reports/grid_returns_swing.parquet (n_days x 27).
"""
from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.strategies.multi_factor_swing import (
    MultiFactorParams,
    run_multi_factor_swing_backtest,
)
from src.utils.logging import configure_logging, get_logger

log = get_logger("build_swing_grid")

TOP_NS = (3, 5, 8)
REBALANCE_DAYS = (5, 10, 21)
SOFT_STOPS = (0.05, 0.08, 0.0)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "build_swing_grid.log")

    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--universe", default="halal")
    p.add_argument("--out", type=Path,
                   default=Path("reports/grid_returns_swing.parquet"))
    args = p.parse_args(argv)

    if args.universe == "halal":
        from config.halal_universe import HALAL_UNIVERSE
        symbols = list(HALAL_UNIVERSE)
    else:
        from config.sp100_universe import SP100_UNIVERSE
        symbols = list(SP100_UNIVERSE)

    start_d = datetime.fromisoformat(args.start).date()
    end_d = datetime.fromisoformat(args.end).date()
    daily_dir = settings.data_dir / "parquet_daily"

    daily_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = daily_dir / f"{sym.upper()}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        if d.empty:
            continue
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        daily_by_sym[sym] = d
    log.info("swing_grid.loaded", extra={"n_symbols": len(daily_by_sym)})

    # LEARNED: pull the earnings cache and pass it into the backtest so
    # the PEAD factor (10% weight in MultiFactorParams) is actually live
    # during CPCV. The previous grid ran with earnings_events=None which
    # forced PEAD to 0.
    from src.signals.pead import load_earnings_events
    earnings_events = load_earnings_events()
    log.info("swing_grid.earnings_loaded",
             extra={"n_events": len(earnings_events)})

    series_by_variant: dict[str, pd.Series] = {}
    for top_n, reb, stop in product(TOP_NS, REBALANCE_DAYS, SOFT_STOPS):
        name = f"top{top_n:02d}_reb{reb:02d}_stop{int(stop*100):02d}"
        cfg = MultiFactorParams(
            initial_equity=settings.initial_equity,
            top_n=top_n, rebalance_days=reb, soft_stop_pct=stop,
        )
        log.info("swing_grid.run", extra={"variant": name})
        result = run_multi_factor_swing_backtest(
            daily_by_sym, start=start_d, end=end_d, params=cfg,
            earnings_events=earnings_events,
        )
        series_by_variant[name] = result.daily_returns
        log.info("swing_grid.variant_done",
                 extra={"variant": name, "n_trades": len(result.trades),
                        "terminal": float(result.equity_curve.iloc[-1]) if len(result.equity_curve) else 0.0})

    df = pd.DataFrame(series_by_variant).fillna(0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    log.info("swing_grid.done",
             extra={"path": str(args.out), "shape": list(df.shape),
                    "variants": list(df.columns)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
