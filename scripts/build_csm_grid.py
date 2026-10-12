"""Build the CSM parameter-grid returns matrix for CPCV.

Sweeps a small, principled grid:
    top_n           in {5, 10, 15}
    rebalance_days  in {15, 21, 30}
    weight_scheme   in {equal, inverse_vol}
    abs_threshold   in {0.0, 0.05}        # raw 12-1 must beat 0% or 5%

The default 18-variant grid balances coverage vs. multiple-testing penalty.

Output: reports/grid_returns.parquet — shape (n_days, n_variants).
"""
from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.strategies.cross_sectional_momentum import CSMParams, run_csm_backtest
from src.utils.logging import configure_logging, get_logger

log = get_logger("build_csm_grid")

TOP_NS = (5, 10, 15)
REBALANCE_DAYS = (15, 21, 30)
WEIGHT_SCHEMES = ("equal", "inverse_vol")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "build_csm_grid.log")

    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--symbols", nargs="+", required=False)
    p.add_argument("--universe", default=None,
                   help="optional: 'sp100' for the curated SP100 list")
    p.add_argument(
        "--out", type=Path, default=Path("reports/grid_returns.parquet")
    )
    args = p.parse_args(argv)

    if args.universe == "sp100":
        from config.sp100_universe import SP100_UNIVERSE
        symbols = list(SP100_UNIVERSE)
    elif args.symbols:
        symbols = list(args.symbols)
    else:
        log.error("csm_grid.no_symbols")
        return 1

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
        d = d.loc[(d.index >= pd.Timestamp(start_d)) & (d.index <= pd.Timestamp(end_d))]
        if d.empty:
            continue
        daily_by_sym[sym] = d
    log.info("csm_grid.loaded", extra={"n_symbols": len(daily_by_sym)})

    series_by_variant: dict[str, pd.Series] = {}
    for top_n, reb, ws in product(TOP_NS, REBALANCE_DAYS, WEIGHT_SCHEMES):
        name = f"top{top_n:02d}_reb{reb:02d}_{ws}"
        cfg = CSMParams(
            initial_equity=settings.initial_equity,
            top_n=top_n, rebalance_days=reb,
            abs_momentum_threshold=0.0,  # always-on TS filter
            weight_scheme=ws,
        )
        log.info("csm_grid.run", extra={"variant": name})
        result = run_csm_backtest(daily_by_sym, start=start_d, end=end_d, params=cfg)
        series_by_variant[name] = result.daily_returns

    df = pd.DataFrame(series_by_variant).fillna(0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    log.info(
        "csm_grid.done",
        extra={"path": str(args.out), "shape": list(df.shape),
               "variants": list(df.columns)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
