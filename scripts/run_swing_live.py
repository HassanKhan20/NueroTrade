"""Live paper-trading runner for the multi-factor swing strategy.

Mirrors run_csm_live.py: idempotent, refuses to fire if rebalance interval
hasn't elapsed, refuses to run live (ALPACA_PAPER must be true), uses
deterministic client_order_ids so retries don't double-fill.

Frozen parameters baked in: replace only via the CPCV harness, never
hand-tune in production.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.live.alpaca_client import AlpacaBrokerClient
from src.strategies.multi_factor_swing import (
    MultiFactorParams,
    rank_universe,
)
from src.utils.logging import configure_logging, get_logger

log = get_logger("run_swing_live")

# Frozen params — set once after CPCV. Don't tune live.
FROZEN_PARAMS = MultiFactorParams(
    top_n=5,
    rebalance_days=10,
    weight_technical=0.40,
    weight_momentum=0.30,
    weight_quality=0.20,
    weight_pead=0.10,
    soft_stop_pct=0.08,
    initial_equity=25_000.0,  # ignored at runtime — uses live equity
    slippage_bps=5.0,
)


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"last_rebalance_date": None, "n_rebalances": 0}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {"last_rebalance_date": None, "n_rebalances": 0}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str))


def _trading_days_between(d1: str | None, d2_iso: str) -> int:
    if not d1:
        return 9999
    a = datetime.fromisoformat(d1).date()
    b = datetime.fromisoformat(d2_iso).date()
    return len(pd.bdate_range(a, b)) - 1


def _load_daily_universe(symbols: list[str], cache_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = cache_dir / f"{sym.upper()}.parquet"
        if not path.exists():
            log.warning("swing_live.no_cache", extra={"symbol": sym})
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        out[sym] = df
    return out


def _compute_target_allocation(
    daily_by_sym: dict[str, pd.DataFrame], equity: float,
    today_iso: str, params: MultiFactorParams,
) -> dict[str, float]:
    """Return symbol -> dollar allocation for today's rebalance."""
    today_d = datetime.fromisoformat(today_iso).date()
    ranked = rank_universe(daily_by_sym, today_d, None, params)
    qualified = [r for r in ranked if r[1] >= params.min_composite_score]
    picks = qualified[: params.top_n]
    if not picks:
        return {}
    per_name = min(equity / len(picks), equity * params.max_position_pct)
    return {sym: per_name for sym, _, _ in picks}


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "swing_live.log")

    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, default=Path("logs/swing_state.json"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    if not settings.alpaca_paper:
        log.error("swing_live.refusing_live_mode")
        return 2
    if not settings.alpaca_key or not settings.alpaca_secret:
        log.error("swing_live.no_credentials")
        return 2

    from config.halal_universe import HALAL_UNIVERSE
    symbols = list(HALAL_UNIVERSE)

    today_iso = datetime.now(timezone.utc).date().isoformat()
    state = _load_state(args.state)
    days_since = _trading_days_between(state.get("last_rebalance_date"), today_iso)
    if days_since < FROZEN_PARAMS.rebalance_days and not args.force:
        log.info("swing_live.skip_rebalance",
                 extra={"days_since": days_since,
                        "interval": FROZEN_PARAMS.rebalance_days,
                        "last": state.get("last_rebalance_date")})
        return 0

    log.info("swing_live.rebalance_due",
             extra={"days_since": days_since, "today": today_iso})

    daily_by_sym = _load_daily_universe(
        symbols, settings.data_dir / "parquet_daily",
    )
    log.info("swing_live.data_loaded",
             extra={"n_symbols": len(daily_by_sym)})

    broker = AlpacaBrokerClient(settings.alpaca_key, settings.alpaca_secret, paper=True)
    acct = broker.account()
    current_positions = {p.symbol: p for p in broker.positions()}
    log.info("swing_live.account",
             extra={"equity": acct.equity, "cash": acct.cash,
                    "n_positions": len(current_positions)})

    targets = _compute_target_allocation(
        daily_by_sym, acct.equity, today_iso, FROZEN_PARAMS,
    )
    if not targets:
        log.warning("swing_live.no_qualified")

    log.info("swing_live.targets",
             extra={"picks": list(targets.keys()), "n_picks": len(targets)})

    orders: list[dict] = []
    # Close anything not in targets
    for sym, pos in current_positions.items():
        if sym not in targets:
            orders.append({"action": "close", "symbol": sym, "qty": int(pos.qty)})
    # Open / adjust
    for sym, dollar_alloc in targets.items():
        df = daily_by_sym.get(sym)
        if df is None or df.empty:
            continue
        last_close = float(df["close"].iloc[-1])
        target_shares = math.floor(dollar_alloc / last_close)
        if target_shares <= 0:
            continue
        cur_qty = int(current_positions[sym].qty) if sym in current_positions else 0
        delta = target_shares - cur_qty
        if delta > 0:
            orders.append({"action": "buy", "symbol": sym, "qty": delta})
        elif delta < 0:
            orders.append({"action": "sell_partial", "symbol": sym,
                           "qty": -delta, "target_qty": target_shares})

    log.info("swing_live.orders_planned", extra={"n_orders": len(orders)})

    if args.dry_run:
        print("=== DRY RUN — no orders submitted ===")
        for o in orders:
            print(f"  {o['action'].upper():>13}  {o['symbol']:<6}  {o['qty']} sh")
        return 0

    n_ok = n_failed = 0
    for o in orders:
        coid = f"swing-{today_iso}-{o['symbol']}-{o['action']}"
        try:
            if o["action"] == "close":
                broker.close_position(o["symbol"])
            elif o["action"] == "buy":
                broker.submit_market_long(o["symbol"], o["qty"], client_order_id=coid)
            elif o["action"] == "sell_partial":
                # Simplest: full close + re-open at target qty
                broker.close_position(o["symbol"])
                if o["target_qty"] > 0:
                    broker.submit_market_long(
                        o["symbol"], o["target_qty"],
                        client_order_id=coid + "-reopen",
                    )
            n_ok += 1
            log.info("swing_live.order_ok", extra={**o, "coid": coid})
        except Exception:  # noqa: BLE001
            n_failed += 1
            log.exception("swing_live.order_failed", extra=o)

    state["last_rebalance_date"] = today_iso
    state["n_rebalances"] = int(state.get("n_rebalances", 0)) + 1
    state["last_picks"] = list(targets.keys())
    state["last_equity"] = acct.equity
    _save_state(args.state, state)

    log.info("swing_live.done",
             extra={"submitted": n_ok, "failed": n_failed,
                    "n_picks": len(targets)})
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
