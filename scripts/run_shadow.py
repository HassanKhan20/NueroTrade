"""Shadow-mode runner — wires the Runner to a ShadowBroker and a single day's
inputs supplied on the command line. Used during the 30-day shadow phase
before paper trading.

The harness deliberately keeps "what is today's session?" outside this file:
the operator passes the date and the data files. That makes it possible to
replay a specific day for diagnostics.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.backtest.engine import DayInputs
from src.data.ingest_alpaca import load_range
from src.live.runner import Runner, RunnerConfig
from src.live.shadow_mode import ShadowBroker
from src.utils.clock import NY_TZ
from src.utils.logging import configure_logging, get_logger

log = get_logger("run_shadow")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "shadow.log")

    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--momentum-symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--vix-symbol", default="^VIX")
    p.add_argument("--starting-equity", type=float, default=25_000.0)
    args = p.parse_args(argv)

    d = datetime.fromisoformat(args.date).date()

    minute_orb: dict[str, pd.DataFrame] = {}
    minute_mom: dict[str, pd.DataFrame] = {}
    daily_orb: dict[str, pd.DataFrame] = {}
    prev_close_mom: dict[str, float] = {}

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
            prev_close_mom[sym] = float(daily["close"].iloc[-1])

    vix_minute = load_range(args.vix_symbol, d, d, settings.data_dir)
    vix_today = float(vix_minute["close"].iloc[-1]) if not vix_minute.empty else float("nan")

    day = DayInputs(
        session_date=d,
        minute_bars_orb=minute_orb,
        daily_bars_orb=daily_orb,
        minute_bars_mom=minute_mom,
        prev_close_mom=prev_close_mom,
        vix_today=vix_today,
        vix_history=pd.Series(dtype=float),
        spy_close_history=pd.Series(dtype=float),
    )

    broker = ShadowBroker(
        log_path=settings.logs_dir / "shadow_orders.jsonl",
        starting_equity=args.starting_equity,
    )
    runner = Runner(broker=broker, cfg=RunnerConfig(state_path=Path("bot_state.json")))
    trades = runner.run_session(day)
    log.info("shadow.done", extra={"n_trades": len(trades)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
