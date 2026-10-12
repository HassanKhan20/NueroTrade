"""CLI entry point: run a backtest on cached data and write a report.

Usage:
    python -m scripts.run_backtest --start 2018-01-01 --end 2023-12-31 \\
        --symbols AAPL MSFT NVDA SPY QQQ --label is

This script intentionally keeps the data-loading / orchestration thin so the
real complexity lives in src/backtest/engine.py.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.backtest.engine import BacktestConfig, Backtester, DayInputs
from src.backtest.reports import write_report
from src.data.ingest_alpaca import load_range
from src.strategies.ict_long_only import ICTParams
from src.strategies.intraday_momentum import IntradayMomentumParams
from src.strategies.orb_stocks_in_play import ORBParams
from src.utils.clock import NY_TZ, trading_days
from src.utils.logging import configure_logging, get_logger

log = get_logger("run_backtest")


def _to_ny(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.tz_convert(NY_TZ)


def _ts(d: date, idx: pd.DatetimeIndex) -> pd.Timestamp:
    return pd.Timestamp(d, tz=idx.tz) if idx.tz is not None else pd.Timestamp(d)


def _slice_before(s: "pd.Series", d: date) -> "pd.Series":
    if s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return s
    return s.loc[s.index < _ts(d, s.index)]


def _last_at_or_before(s: "pd.Series", d: date) -> float:
    if s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return float("nan")
    sub = s.loc[: _ts(d, s.index)]
    return float(sub.iloc[-1]) if not sub.empty else float("nan")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "backtest.log")

    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--momentum-symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--vix-symbol", default="^VIX")
    p.add_argument("--label", default="run")
    # ORB tunables
    p.add_argument("--orb-inverted", action="store_true",
                   help="ORB enters on RED opening-range candles instead of green")
    p.add_argument("--orb-take-profit-r", type=float, default=0.0,
                   help="ORB profit target as multiple of stop distance (0=none)")
    p.add_argument("--orb-stop-atr-pct", type=float, default=0.10)
    p.add_argument("--orb-top-n", type=int, default=20)
    # Intraday-momentum tunables
    p.add_argument("--mom-inverted", action="store_true",
                   help="Intraday momentum enters on both-negative instead of both-positive")
    p.add_argument("--mom-take-profit-multiple", type=float, default=0.0,
                   help="Intraday momentum TP as multiple of expected_move (0=none)")
    # ICT mode (replaces ORB + momentum when enabled)
    p.add_argument("--use-ict", action="store_true",
                   help="Run the ICT multi-timeframe alignment strategy "
                        "instead of ORB + intraday momentum")
    p.add_argument("--ict-take-profit-r", type=float, default=2.0,
                   help="ICT take profit as a multiple of stop distance (default 2R)")
    p.add_argument("--ict-htf-min-pct", type=float, default=0.001,
                   help="Minimum 1H FVG width as a fraction of price (default 0.10%%)")
    p.add_argument("--ict-mtf-min-pct", type=float, default=0.0005,
                   help="Minimum 5M FVG width as a fraction of price (default 0.05%%)")
    p.add_argument(
        "--ict-bias-mode",
        choices=["qqq", "self", "qqq_or_self", "daily_self", "none"],
        default="qqq",
        help=(
            "Bias filter for ICT: 'qqq' (default) requires QQQ uptrend; "
            "'self' requires symbol intraday uptrend (5-min EMA); "
            "'qqq_or_self' allows either; "
            "'daily_self' requires symbol DAILY 50-SMA > 200-SMA (most permissive); "
            "'none' disables the bias filter."
        ),
    )
    args = p.parse_args(argv)

    start_d = datetime.fromisoformat(args.start).date()
    end_d = datetime.fromisoformat(args.end).date()

    log.info("backtest.load_start", extra={"start": str(start_d), "end": str(end_d)})

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
            # Strip tz from the daily index — the strategies compare against
            # naive pd.Timestamp(session_date) values.
            if d.index.tz is not None:
                d.index = d.index.tz_localize(None)
            daily_by_sym[sym] = d

    spy = daily_by_sym.get("SPY", pd.DataFrame())
    spy_close_history = spy["close"] if not spy.empty else pd.Series(dtype=float)
    vix_minute = load_range(args.vix_symbol, start_d, end_d, settings.data_dir)
    if not vix_minute.empty:
        vix_daily = vix_minute["close"].resample("1D").last().dropna()
        if vix_daily.index.tz is not None:
            vix_daily.index = vix_daily.index.tz_localize(None)
    else:
        vix_daily = pd.Series(dtype=float)

    days_inputs: list[DayInputs] = []
    for d in trading_days(start_d, end_d):
        # ORB universe screening needs ~14 days of prior 09:30-09:35 volume,
        # so we pass FULL history per symbol. The fill simulator clips to
        # session_date internally to avoid future-bar leak.
        minute_orb = {
            sym: minute_by_sym[sym]
            for sym in args.symbols
            if sym in minute_by_sym
        }
        minute_mom = {
            sym: minute_by_sym[sym]
            for sym in args.momentum_symbols
            if sym in minute_by_sym
        }
        prev_close_mom: dict[str, float] = {}
        for sym in args.momentum_symbols:
            ds = daily_by_sym.get(sym, pd.DataFrame())
            if ds.empty or not isinstance(ds.index, pd.DatetimeIndex):
                continue
            cutoff = pd.Timestamp(d, tz=ds.index.tz) if ds.index.tz is not None else pd.Timestamp(d)
            prior = ds.loc[ds.index < cutoff]
            if not prior.empty:
                prev_close_mom[sym] = float(prior["close"].iloc[-1])

        # ICT mode also wants the QQQ 1-min frame for the bias filter.
        qqq_full = minute_by_sym.get("QQQ", pd.DataFrame())

        days_inputs.append(
            DayInputs(
                session_date=d,
                minute_bars_orb=minute_orb,
                daily_bars_orb={s: daily_by_sym.get(s, pd.DataFrame()) for s in args.symbols},
                minute_bars_mom=minute_mom,
                prev_close_mom=prev_close_mom,
                vix_today=_last_at_or_before(vix_daily, d),
                vix_history=_slice_before(vix_daily, d),
                spy_close_history=_slice_before(spy_close_history, d),
                qqq_1min=qqq_full,
            )
        )

    cfg = BacktestConfig(
        initial_equity=settings.initial_equity,
        orb_params=ORBParams(
            top_n=args.orb_top_n,
            stop_atr_pct=args.orb_stop_atr_pct,
            take_profit_r=args.orb_take_profit_r,
            inverted=args.orb_inverted,
        ),
        momentum_params=IntradayMomentumParams(
            inverted=args.mom_inverted,
            take_profit_multiple=args.mom_take_profit_multiple,
        ),
        ict_params=ICTParams(
            take_profit_r=args.ict_take_profit_r,
            htf_fvg_min_pct=args.ict_htf_min_pct,
            mtf_fvg_min_pct=args.ict_mtf_min_pct,
            bias_mode=args.ict_bias_mode,
        ),
        enable_ict=args.use_ict,
        progress_label=args.label,
    )
    bt = Backtester(cfg)
    log.info("backtest.run", extra={"n_days": len(days_inputs)})
    result = bt.run(days_inputs)

    out_dir = settings.reports_dir / "backtests" / f"{args.label}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    summary_path = write_report(result, out_dir, label=args.label)
    log.info("backtest.done", extra={"summary": str(summary_path), "trades": len(result.trades)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
