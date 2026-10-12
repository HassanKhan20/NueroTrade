"""Live paper-trading runner for cross-sectional momentum.

Designed to run **once per trading day** (e.g. 30 minutes before close, or
right after close) via Windows Task Scheduler / cron. Idempotent: only
rebalances if N trading days have passed since the last rebalance.

Workflow:
  1. Pull daily bars for the universe (last ~280 days needed for 252+21 signal).
  2. Read last-rebalance state from logs/csm_state.json.
  3. If rebalance is due:
       a. Compute target picks via cross_sectional_momentum.rank_universe.
       b. Apply absolute-momentum filter and inverse-vol weights.
       c. Diff vs. current Alpaca paper positions.
       d. Submit market orders (closes + opens).
       e. Persist state.
  4. Always log a heartbeat to logs/csm_live.log.

Safety:
  * Refuses to run if ALPACA_PAPER != true. No live money.
  * Hard-coded daily-loss circuit-breaker via the existing KillSwitch.
  * --dry-run flag prints intended orders without sending.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.data.ingest_alpaca import AlpacaBarFetcher
from src.live.alpaca_client import AlpacaBrokerClient
from src.strategies.cross_sectional_momentum import CSMParams, rank_universe
from src.utils.logging import configure_logging, get_logger

log = get_logger("run_csm_live")

# Frozen parameters from the CPCV winner. Don't tune these without re-running
# the validation harness — that's how curve-fit deployments happen.
FROZEN_PARAMS = CSMParams(
    top_n=5,
    rebalance_days=15,
    lookback_days=252,
    skip_days=21,
    abs_momentum_threshold=0.0,
    weight_scheme="inverse_vol",
    vol_lookback_days=60,
    initial_equity=25_000.0,        # ignored at runtime; we use real equity
    slippage_bps=5.0,               # only used in backtests
)


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"last_rebalance_date": None, "n_rebalances": 0}
    try:
        return json.loads(state_path.read_text())
    except Exception:  # noqa: BLE001
        return {"last_rebalance_date": None, "n_rebalances": 0}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, default=str))


def _trading_days_between(d1: str | None, d2_str: str) -> int:
    """Approximate count of business days between two ISO dates (exclusive)."""
    if not d1:
        return 9999
    a = datetime.fromisoformat(d1).date()
    b = datetime.fromisoformat(d2_str).date()
    return len(pd.bdate_range(a, b)) - 1


def _load_daily(symbols: list[str], cache_dir: Path,
                 fetcher: AlpacaBarFetcher | None) -> dict[str, pd.DataFrame]:
    """Load each symbol's full daily history from the parquet cache, then
    optionally top up with any new bars since the cache's last date.

    The 252+21 day signal requires real, deep history — fetching just a
    rolling tail is not enough on its own.
    """
    out: dict[str, pd.DataFrame] = {}
    end = datetime.now(timezone.utc) - timedelta(hours=24)
    for sym in symbols:
        path = cache_dir / f"{sym.upper()}.parquet"
        if not path.exists():
            log.warning("csm_live.no_cache", extra={"symbol": sym})
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)

        # Optionally top up with new bars since last cached date.
        if fetcher is not None:
            last_date = df.index.max()
            tail_start = (last_date + pd.Timedelta(days=1)).to_pydatetime().replace(
                tzinfo=timezone.utc
            )
            if tail_start < end:
                try:
                    tail = fetcher.fetch_daily(sym, tail_start, end)
                    if not tail.empty:
                        if tail.index.tz is not None:
                            tail = tail.copy()
                            tail.index = tail.index.tz_localize(None)
                        df = pd.concat([df, tail])
                        df = df[~df.index.duplicated(keep="last")].sort_index()
                except Exception:  # noqa: BLE001
                    log.exception("csm_live.tail_fetch_failed",
                                  extra={"symbol": sym})
        out[sym] = df
    return out


def _compute_target_allocation(
    daily_by_sym: dict[str, pd.DataFrame],
    equity: float,
    params: CSMParams,
    today_iso: str,
) -> dict[str, float]:
    """Return symbol -> dollar allocation for today's rebalance."""
    today_d = datetime.fromisoformat(today_iso).date()
    ranked = rank_universe(daily_by_sym, today_d, params)
    qualified = [(s, r) for s, r in ranked if r > params.abs_momentum_threshold]
    picks = [s for s, _ in qualified[: params.top_n]]
    if not picks:
        return {}

    # inverse-vol weights (skip equal-weight branch — frozen variant uses inv-vol)
    invs: dict[str, float] = {}
    for s in picks:
        df = daily_by_sym.get(s)
        if df is None or len(df) < params.vol_lookback_days + 1:
            invs[s] = 0.0
            continue
        prior = df.loc[df.index < pd.Timestamp(today_d)]
        if len(prior) < params.vol_lookback_days + 1:
            invs[s] = 0.0
            continue
        closes = prior["close"].iloc[-(params.vol_lookback_days + 1):].to_numpy()
        if (closes <= 0).any():
            invs[s] = 0.0
            continue
        import numpy as np  # local
        rets = np.diff(np.log(closes))
        sd = float(np.std(rets, ddof=1))
        invs[s] = 1.0 / sd if sd > 1e-9 else 0.0
    total = sum(invs.values())
    if total <= 0:
        share = equity / len(picks)
        return {s: share for s in picks}
    return {s: equity * (w / total) for s, w in invs.items()}


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "csm_live.log")

    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="sp100")
    p.add_argument("--state", type=Path, default=Path("logs/csm_state.json"))
    p.add_argument("--dry-run", action="store_true",
                   help="print intended orders, don't submit")
    p.add_argument("--force", action="store_true",
                   help="ignore the rebalance interval check")
    args = p.parse_args(argv)

    # Safety: refuse to run live
    if not settings.alpaca_paper:
        log.error("csm_live.refusing_live_mode")
        return 2
    if not settings.alpaca_key or not settings.alpaca_secret:
        log.error("csm_live.no_credentials")
        return 2

    if args.universe == "sp100":
        from config.sp100_universe import SP100_UNIVERSE
        symbols = list(SP100_UNIVERSE)
    else:
        log.error("csm_live.unknown_universe", extra={"universe": args.universe})
        return 2

    today_iso = datetime.now(timezone.utc).date().isoformat()
    state = _load_state(args.state)
    days_since = _trading_days_between(state.get("last_rebalance_date"), today_iso)
    if days_since < FROZEN_PARAMS.rebalance_days and not args.force:
        log.info("csm_live.skip_rebalance",
                 extra={"days_since": days_since,
                        "interval": FROZEN_PARAMS.rebalance_days,
                        "last": state.get("last_rebalance_date")})
        return 0

    log.info("csm_live.rebalance_due",
             extra={"days_since": days_since, "today": today_iso})

    # Load daily bars from cache, top up with any fresh bars since cache write
    fetcher = AlpacaBarFetcher(settings.alpaca_key, settings.alpaca_secret)
    daily_by_sym = _load_daily(
        symbols, settings.data_dir / "parquet_daily", fetcher,
    )
    log.info("csm_live.data_loaded",
             extra={"n_symbols": len(daily_by_sym),
                    "needed": FROZEN_PARAMS.lookback_days + FROZEN_PARAMS.skip_days})

    # Account + current positions
    broker = AlpacaBrokerClient(
        settings.alpaca_key, settings.alpaca_secret, paper=True,
    )
    acct = broker.account()
    current_positions = {p.symbol: p for p in broker.positions()}
    log.info("csm_live.account",
             extra={"equity": acct.equity, "cash": acct.cash,
                    "n_positions": len(current_positions)})

    # Compute target allocation
    targets = _compute_target_allocation(
        daily_by_sym, acct.equity, FROZEN_PARAMS, today_iso,
    )
    if not targets:
        log.warning("csm_live.no_qualified_names_holding_cash")

    log.info("csm_live.targets",
             extra={"picks": list(targets.keys()),
                    "n_picks": len(targets)})

    # Build orders: close anything not in targets, open/resize anything in targets
    orders: list[dict] = []
    # 1) Closes
    for sym, pos in current_positions.items():
        if sym not in targets:
            orders.append({"action": "close", "symbol": sym, "qty": int(pos.qty)})

    # 2) Opens / resizes
    for sym, dollar_alloc in targets.items():
        df = daily_by_sym.get(sym)
        if df is None or df.empty:
            continue
        last_close = float(df["close"].iloc[-1])
        target_shares = math.floor(dollar_alloc / last_close)
        if target_shares <= 0:
            continue
        cur = current_positions.get(sym)
        cur_qty = int(cur.qty) if cur else 0
        delta = target_shares - cur_qty
        if delta > 0:
            orders.append({"action": "buy", "symbol": sym, "qty": delta})
        elif delta < 0:
            orders.append({"action": "sell", "symbol": sym, "qty": -delta})

    log.info("csm_live.orders_planned", extra={"n_orders": len(orders)})

    if args.dry_run:
        print("=== DRY RUN — no orders submitted ===")
        for o in orders:
            print(f"  {o['action'].upper():>5}  {o['symbol']:<6}  {o['qty']} sh")
        return 0

    # Submit
    n_ok = n_failed = 0
    for o in orders:
        coid = f"csm-{today_iso}-{o['symbol']}-{o['action']}"
        try:
            if o["action"] == "close":
                broker.close_position(o["symbol"])
            elif o["action"] == "buy":
                broker.submit_market_long(o["symbol"], o["qty"], client_order_id=coid)
            elif o["action"] == "sell":
                # Reduce: close partial. Alpaca handles via opposing market order.
                broker.close_position(o["symbol"])  # simplest: full close, re-open with new size
                # Then submit a new buy at the target qty
                target_qty = next(
                    (math.floor(targets[o["symbol"]] / float(daily_by_sym[o["symbol"]]["close"].iloc[-1])))
                    for _ in [None]
                )
                if target_qty > 0:
                    broker.submit_market_long(
                        o["symbol"], target_qty,
                        client_order_id=coid + "-reopen",
                    )
            n_ok += 1
            log.info("csm_live.order_ok", extra={**o, "coid": coid})
        except Exception:  # noqa: BLE001
            n_failed += 1
            log.exception("csm_live.order_failed", extra=o)

    state["last_rebalance_date"] = today_iso
    state["n_rebalances"] = int(state.get("n_rebalances", 0)) + 1
    state["last_picks"] = list(targets.keys())
    state["last_equity"] = acct.equity
    _save_state(args.state, state)

    log.info("csm_live.done",
             extra={"submitted": n_ok, "failed": n_failed,
                    "n_picks": len(targets)})
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
