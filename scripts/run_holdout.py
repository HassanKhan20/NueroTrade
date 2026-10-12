"""Holdout gate: run frozen params on the untouched 2024-2025 window and
verify pass-criteria.

Hard fails (non-zero exit) if:
    * params YAML is not frozen
    * OOS Sharpe < 0.7 * IS Sharpe
    * Max drawdown > 1.5 * IS max drawdown
    * Worst single month < -12%
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.utils.logging import configure_logging, get_logger

log = get_logger("run_holdout")

OOS_SHARPE_RATIO_GATE = 0.70
MAX_DD_MULTIPLE_GATE = 1.50
WORST_MONTH_GATE = -0.12


def main(argv: list[str] | None = None) -> int:
    configure_logging("INFO")
    p = argparse.ArgumentParser()
    p.add_argument("--params", type=Path, default=Path("config/strategy_params.yaml"))
    p.add_argument("--is-summary", type=Path, required=True,
                   help="path to in-sample backtest summary.json")
    p.add_argument("--oos-summary", type=Path, required=True,
                   help="path to OOS (holdout) backtest summary.json")
    p.add_argument("--oos-equity", type=Path, required=True,
                   help="path to OOS equity_curve.csv")
    args = p.parse_args(argv)

    params = yaml.safe_load(args.params.read_text())
    if not params.get("frozen"):
        log.error("holdout.params_not_frozen")
        return 1

    is_sum = json.loads(args.is_summary.read_text())
    oos_sum = json.loads(args.oos_summary.read_text())

    is_sharpe = is_sum["sharpe"]
    oos_sharpe = oos_sum["sharpe"]
    is_dd = abs(is_sum["max_drawdown"])
    oos_dd = abs(oos_sum["max_drawdown"])

    eq = pd.read_csv(args.oos_equity, index_col=0, parse_dates=True)["equity"]
    monthly = eq.resample("ME").last().pct_change().dropna()
    worst_month = float(monthly.min()) if len(monthly) else 0.0

    failures: list[str] = []
    if is_sharpe > 0 and oos_sharpe < OOS_SHARPE_RATIO_GATE * is_sharpe:
        failures.append(
            f"OOS sharpe {oos_sharpe:.3f} < {OOS_SHARPE_RATIO_GATE} * IS {is_sharpe:.3f}"
        )
    if is_dd > 0 and oos_dd > MAX_DD_MULTIPLE_GATE * is_dd:
        failures.append(
            f"OOS max DD {oos_dd:.4f} > {MAX_DD_MULTIPLE_GATE} * IS {is_dd:.4f}"
        )
    if worst_month < WORST_MONTH_GATE:
        failures.append(
            f"Worst OOS month {worst_month:.4f} < {WORST_MONTH_GATE}"
        )

    if failures:
        for f in failures:
            log.error("holdout.fail", extra={"reason": f})
        return 3
    log.info(
        "holdout.pass",
        extra={
            "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe,
            "is_dd": is_dd, "oos_dd": oos_dd, "worst_month": worst_month,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
