"""Build the parameter-grid returns matrix consumed by run_cpcv.

For each ORBParams variant in a small, principled grid, we run a backtest
over the same in-sample window, and stack the daily-returns series into a
single wide Parquet of shape (n_days, n_variants).

The grid is *deliberately small* — large grids inflate the multiple-testing
penalty in PBO and DSR. We sweep two axes:
   * stop_atr_pct  in {0.08, 0.10, 0.12, 0.15}
   * top_n         in {10, 20}
That's 8 trials, which is what the harness was tuned for.

Usage:
    python -m scripts.build_grid --start 2016-01-01 --end 2023-12-31 \\
        --symbols AAPL MSFT NVDA ... SPY QQQ
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.backtest.engine import BacktestConfig, Backtester, DayInputs
from src.data.ingest_alpaca import load_range
from src.strategies.orb_stocks_in_play import ORBParams
from src.utils.clock import NY_TZ, trading_days
from src.utils.logging import configure_logging, get_logger

log = get_logger("build_grid")


# Keep the grid small: 4 stop multiples * 2 top_n = 8 variants
STOP_ATR_PCTS = (0.08, 0.10, 0.12, 0.15)
TOP_NS = (10, 20)


def _slice_day(bars: pd.DataFrame, d: date) -> pd.DataFrame:
    if bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
        return bars
    start = pd.Timestamp(d, tz=NY_TZ)
    end = pd.Timestamp(d + timedelta(days=1), tz=NY_TZ)
    return bars[(bars.index >= start) & (bars.index < end)]


def _to_ny(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.tz_convert(NY_TZ)


def _build_day_inputs(
    days: list[date],
    symbols: list[str],
    momentum_symbols: list[str],
    vix_symbol: str,
    data_dir: Path,
    start_d: date,
    end_d: date,
) -> list[DayInputs]:
    minute_by_sym: dict[str, pd.DataFrame] = {}
    daily_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols + momentum_symbols:
        m = load_range(sym, start_d, end_d, data_dir)
        minute_by_sym[sym] = _to_ny(m)
        if m.empty:
            daily_by_sym[sym] = pd.DataFrame()
        else:
            daily_by_sym[sym] = (
                m.resample("1D")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
            )

    spy = daily_by_sym.get("SPY", pd.DataFrame())
    spy_close = spy["close"] if not spy.empty else pd.Series(dtype=float)
    vix_min = load_range(vix_symbol, start_d, end_d, data_dir)
    vix_daily = (
        vix_min["close"].resample("1D").last().dropna()
        if not vix_min.empty
        else pd.Series(dtype=float)
    )

    def _slice_before(s: pd.Series, d: date) -> pd.Series:
        if s.empty or not isinstance(s.index, pd.DatetimeIndex):
            return s
        return s.loc[s.index < pd.Timestamp(d)]

    def _last_at_or_before(s: pd.Series, d: date) -> float:
        if s.empty or not isinstance(s.index, pd.DatetimeIndex):
            return float("nan")
        sub = s.loc[: pd.Timestamp(d)]
        return float(sub.iloc[-1]) if not sub.empty else float("nan")

    out: list[DayInputs] = []
    for d in days:
        minute_orb = {s: _slice_day(minute_by_sym[s], d) for s in symbols if s in minute_by_sym}
        minute_mom = {
            s: _slice_day(minute_by_sym[s], d) for s in momentum_symbols if s in minute_by_sym
        }
        prev_close: dict[str, float] = {}
        for s in momentum_symbols:
            ds = daily_by_sym.get(s, pd.DataFrame())
            if ds.empty or not isinstance(ds.index, pd.DatetimeIndex):
                continue
            prior = ds.loc[ds.index < pd.Timestamp(d)]
            if not prior.empty:
                prev_close[s] = float(prior["close"].iloc[-1])
        out.append(
            DayInputs(
                session_date=d,
                minute_bars_orb=minute_orb,
                daily_bars_orb={s: daily_by_sym.get(s, pd.DataFrame()) for s in symbols},
                minute_bars_mom=minute_mom,
                prev_close_mom=prev_close,
                vix_today=_last_at_or_before(vix_daily, d),
                vix_history=_slice_before(vix_daily, d),
                spy_close_history=_slice_before(spy_close, d),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "build_grid.log")

    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--momentum-symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--vix-symbol", default="VIXY")
    p.add_argument(
        "--out", type=Path, default=Path("reports/grid_returns.parquet")
    )
    args = p.parse_args(argv)

    start_d = datetime.fromisoformat(args.start).date()
    end_d = datetime.fromisoformat(args.end).date()
    days = trading_days(start_d, end_d)
    log.info("grid.load", extra={"n_days": len(days)})

    day_inputs = _build_day_inputs(
        days,
        args.symbols,
        args.momentum_symbols,
        args.vix_symbol,
        settings.data_dir,
        start_d,
        end_d,
    )

    base_cfg = BacktestConfig(initial_equity=settings.initial_equity)
    series_by_variant: dict[str, pd.Series] = {}
    for stop_pct, top_n in product(STOP_ATR_PCTS, TOP_NS):
        name = f"stop{int(stop_pct * 100):02d}_top{top_n:02d}"
        cfg = BacktestConfig(
            initial_equity=base_cfg.initial_equity,
            orb_params=replace(base_cfg.orb_params, stop_atr_pct=stop_pct, top_n=top_n),
            momentum_params=base_cfg.momentum_params,
            vol_regime_params=base_cfg.vol_regime_params,
            allocator_params=base_cfg.allocator_params,
        )
        log.info("grid.run", extra={"variant": name})
        result = Backtester(cfg).run(day_inputs)
        series_by_variant[name] = result.daily_returns

    df = pd.DataFrame(series_by_variant).fillna(0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    log.info(
        "grid.done",
        extra={"path": str(args.out), "shape": list(df.shape), "variants": list(df.columns)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
