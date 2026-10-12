"""Live (or paper) runner — drives one trading session against Alpaca.

This script gets pointed at real Alpaca credentials and run on the operator's
schedule (e.g. via cron at 09:35 NY). Fails closed: if params are not frozen,
exits non-zero without sending any orders.

Use ALPACA_PAPER=true (default) for paper. Set false for real money.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

from config.settings import get_settings
from src.backtest.engine import DayInputs
from src.data.ingest_alpaca import AlpacaBarFetcher, load_range
from src.live.alpaca_client import AlpacaBrokerClient
from src.live.runner import Runner, RunnerConfig
from src.utils.clock import NY_TZ
from src.utils.logging import configure_logging, get_logger

log = get_logger("run_live")


def _require_frozen_params(path: Path) -> dict:
    if not path.exists():
        log.error("live.no_params")
        raise SystemExit(1)
    p = yaml.safe_load(path.read_text())
    if not p.get("frozen"):
        log.error("live.params_not_frozen")
        raise SystemExit(1)
    return p


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "live.log")

    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD; defaults to today (NY)")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--momentum-symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--vix-symbol", default="^VIX")
    p.add_argument(
        "--params", type=Path, default=Path("config/strategy_params.yaml")
    )
    p.add_argument(
        "--state-path", type=Path, default=Path("bot_state.json")
    )
    args = p.parse_args(argv)

    _require_frozen_params(args.params)

    d = (
        datetime.fromisoformat(args.date).date()
        if args.date
        else datetime.now(NY_TZ).date()
    )

    minute_orb, minute_mom, daily_orb, prev_close_mom = {}, {}, {}, {}
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

    vix = load_range(args.vix_symbol, d, d, settings.data_dir)
    vix_today = float(vix["close"].iloc[-1]) if not vix.empty else float("nan")

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

    broker = AlpacaBrokerClient(
        key=settings.alpaca_key,
        secret=settings.alpaca_secret,
        paper=settings.alpaca_paper,
    )
    runner = Runner(
        broker=broker,
        cfg=RunnerConfig(
            state_path=args.state_path,
            daily_loss_limit=settings.daily_loss_limit,
            weekly_loss_limit=settings.weekly_loss_limit,
        ),
    )
    log.info(
        "live.start",
        extra={"date": d.isoformat(), "paper": settings.alpaca_paper},
    )
    submitted = runner.run_session(day)
    log.info("live.session_done", extra={"n_trades": len(submitted)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
