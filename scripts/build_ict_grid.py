"""Build the ICT parameter-grid returns matrix for CPCV.

Sweeps a small, principled grid:
    bias_mode      in {qqq, self, daily_self}
    take_profit_r  in {1.5, 2.0, 3.0}
That's 9 variants. Keep the grid small so the multiple-testing penalty in
PBO / DSR remains modest.

Output: reports/grid_returns.parquet of shape (n_days, n_variants).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, datetime
from itertools import product
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.backtest.engine import BacktestConfig, Backtester, DayInputs
from src.data.ingest_alpaca import load_range
from src.strategies.ict_long_only import ICTParams
from src.utils.clock import NY_TZ, trading_days
from src.utils.logging import configure_logging, get_logger

log = get_logger("build_ict_grid")

BIAS_MODES = ("qqq", "self", "daily_self")
TP_RS = (1.5, 2.0, 3.0)


def _to_ny(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.tz_convert(NY_TZ)


def _slice_before(s: pd.Series, d: date) -> pd.Series:
    if s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return s
    cutoff = pd.Timestamp(d, tz=s.index.tz) if s.index.tz is not None else pd.Timestamp(d)
    return s.loc[s.index < cutoff]


def _last_at_or_before(s: pd.Series, d: date) -> float:
    if s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return float("nan")
    cutoff = pd.Timestamp(d, tz=s.index.tz) if s.index.tz is not None else pd.Timestamp(d)
    sub = s.loc[:cutoff]
    return float(sub.iloc[-1]) if not sub.empty else float("nan")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "build_ict_grid.log")

    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--momentum-symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--vix-symbol", default="VIXY")
    p.add_argument("--out", type=Path, default=Path("reports/grid_returns.parquet"))
    args = p.parse_args(argv)

    start_d = datetime.fromisoformat(args.start).date()
    end_d = datetime.fromisoformat(args.end).date()
    days = trading_days(start_d, end_d)
    log.info("ict_grid.load", extra={"n_days": len(days)})

    minute_by_sym: dict[str, pd.DataFrame] = {}
    daily_by_sym: dict[str, pd.DataFrame] = {}
    for sym in args.symbols + args.momentum_symbols:
        m = load_range(sym, start_d, end_d, settings.data_dir)
        minute_by_sym[sym] = _to_ny(m)
        if m.empty:
            daily_by_sym[sym] = pd.DataFrame()
        else:
            d = m.resample("1D").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
            if d.index.tz is not None:
                d.index = d.index.tz_localize(None)
            daily_by_sym[sym] = d

    spy = daily_by_sym.get("SPY", pd.DataFrame())
    spy_close_history = spy["close"] if not spy.empty else pd.Series(dtype=float)
    vix_min = load_range(args.vix_symbol, start_d, end_d, settings.data_dir)
    if not vix_min.empty:
        vix_daily = vix_min["close"].resample("1D").last().dropna()
        if vix_daily.index.tz is not None:
            vix_daily.index = vix_daily.index.tz_localize(None)
    else:
        vix_daily = pd.Series(dtype=float)
    qqq_full = minute_by_sym.get("QQQ", pd.DataFrame())

    day_inputs: list[DayInputs] = []
    for d in days:
        prev_close: dict[str, float] = {}
        for sym in args.momentum_symbols:
            ds = daily_by_sym.get(sym, pd.DataFrame())
            if ds.empty or not isinstance(ds.index, pd.DatetimeIndex):
                continue
            cutoff = pd.Timestamp(d, tz=ds.index.tz) if ds.index.tz is not None else pd.Timestamp(d)
            prior = ds.loc[ds.index < cutoff]
            if not prior.empty:
                prev_close[sym] = float(prior["close"].iloc[-1])
        day_inputs.append(
            DayInputs(
                session_date=d,
                minute_bars_orb={s: minute_by_sym[s] for s in args.symbols if s in minute_by_sym},
                daily_bars_orb={s: daily_by_sym.get(s, pd.DataFrame()) for s in args.symbols},
                minute_bars_mom={s: minute_by_sym[s] for s in args.momentum_symbols if s in minute_by_sym},
                prev_close_mom=prev_close,
                vix_today=_last_at_or_before(vix_daily, d),
                vix_history=_slice_before(vix_daily, d),
                spy_close_history=_slice_before(spy_close_history, d),
                qqq_1min=qqq_full,
            )
        )

    series_by_variant: dict[str, pd.Series] = {}
    for bias, tpr in product(BIAS_MODES, TP_RS):
        name = f"{bias}_tp{int(tpr*10):02d}"
        cfg = BacktestConfig(
            initial_equity=settings.initial_equity,
            ict_params=ICTParams(bias_mode=bias, take_profit_r=tpr),
            enable_ict=True,
            progress_label=f"grid_{name}",
        )
        log.info("ict_grid.run", extra={"variant": name})
        result = Backtester(cfg).run(day_inputs)
        series_by_variant[name] = result.daily_returns
        log.info(
            "ict_grid.variant_done",
            extra={
                "variant": name,
                "n_trades": len(result.trades),
                "terminal_equity": float(result.equity_curve.iloc[-1]) if len(result.equity_curve) else 0.0,
            },
        )

    df = pd.DataFrame(series_by_variant).fillna(0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    log.info(
        "ict_grid.done",
        extra={"path": str(args.out), "shape": list(df.shape), "variants": list(df.columns)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
