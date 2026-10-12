"""Backtest report writer.

Outputs a JSON summary + a small HTML page with the equity curve. Kept
intentionally minimal — the user said no markdown decoration; we ship
machine-readable artefacts and one chart.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.metrics import (
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)


def summarise(result: BacktestResult) -> dict[str, Any]:
    rets = result.daily_returns.to_numpy()
    pnl = [t.pnl_net for t in result.trades]
    eq = result.equity_curve
    if len(eq) == 0:
        terminal = 0.0
        total_return = 0.0
    else:
        terminal = float(eq.iloc[-1])
        initial = float(eq.iloc[0])
        total_return = (terminal / initial) - 1.0 if initial > 0 else 0.0
    return {
        "n_days": int(len(eq)),
        "n_trades": int(len(result.trades)),
        "terminal_equity": terminal,
        "total_return": total_return,
        "sharpe": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(eq.to_numpy()),
        "profit_factor": profit_factor(pnl),
        "win_rate": win_rate(pnl),
        "component_attribution": result.component_attribution,
    }


def write_report(result: BacktestResult, out_dir: Path, *, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(result)
    summary["label"] = label
    summary["written_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    result.equity_curve.to_csv(out_dir / "equity_curve.csv", header=["equity"])
    pd.DataFrame(
        [
            {
                "symbol": t.symbol,
                "component": t.component,
                "entry_time": t.intent.entry_time,
                "fill_price": t.fill_price,
                "exit_price": t.exit_price,
                "shares": t.intent.shares,
                "exit_reason": t.exit_reason,
                "pnl_gross": t.pnl_gross,
                "commission": t.commission,
                "pnl_net": t.pnl_net,
            }
            for t in result.trades
        ]
    ).to_csv(out_dir / "trades.csv", index=False)
    return out_dir / "summary.json"
