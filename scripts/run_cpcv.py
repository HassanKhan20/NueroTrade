"""CPCV harness: score a parameter grid with PBO + DSR gates.

Returns exit code 0 only if PBO < 0.15 and DSR > 0.95 at the median best
strategy across splits — the canonical "deploy" gate.

Usage:
    python -m scripts.run_cpcv --returns reports/grid_returns.parquet \\
        --metric sharpe --groups 6 --test 2 --embargo 5
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.cpcv import cpcv_splits, score_grid
from src.backtest.metrics import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from src.utils.logging import configure_logging, get_logger

log = get_logger("run_cpcv")

PBO_GATE = 0.15
DSR_GATE = 0.95


def main(argv: list[str] | None = None) -> int:
    configure_logging("INFO")
    p = argparse.ArgumentParser()
    p.add_argument("--returns", type=Path, required=True,
                   help="Parquet of shape (n_obs, n_strategies)")
    p.add_argument("--out", type=Path, default=Path("reports/cpcv"))
    p.add_argument("--groups", type=int, default=6)
    p.add_argument("--test", type=int, default=2)
    p.add_argument("--embargo", type=int, default=5)
    args = p.parse_args(argv)

    rets_df = pd.read_parquet(args.returns)
    rets = {col: rets_df[col].to_numpy(dtype=np.float64) for col in rets_df.columns}
    n_obs = len(rets_df)

    splits = cpcv_splits(
        n_obs, n_groups=args.groups, n_test_groups=args.test, embargo=args.embargo
    )
    is_perf, oos_perf, names = score_grid(rets, splits, metric_fn=sharpe_ratio)
    pbo = probability_of_backtest_overfitting(is_perf, oos_perf)

    # DSR at the strategy with the highest median IS Sharpe.
    median_is = np.median(is_perf, axis=0)
    best_idx = int(np.argmax(median_is))
    best_name = names[best_idx]
    best_returns = rets[best_name]
    sr = sharpe_ratio(best_returns)
    skew = float(pd.Series(best_returns).skew())
    kurt = float(pd.Series(best_returns).kurt() + 3.0)
    # DSR units must all be per-period (Bailey-LdP convention). The Sharpe
    # values returned by sharpe_ratio() are annualised (× sqrt(252)); convert
    # both sr AND trial_sr_var to per-period before the call. Variance scales
    # linearly with the annualisation factor, so divide by 252.
    sr_per_period = sr / np.sqrt(252.0)
    trial_var_annual = float(np.var(median_is, ddof=1)) if len(median_is) > 1 else 1.0
    trial_var_per_period = trial_var_annual / 252.0
    dsr = deflated_sharpe_ratio(
        sr_per_period,
        n=len(best_returns),
        n_trials=len(names),
        skew=skew,
        kurt=kurt,
        trial_sr_var=trial_var_per_period,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_obs": n_obs,
        "n_strategies": len(names),
        "n_splits": len(splits),
        "pbo": pbo,
        "pbo_gate": PBO_GATE,
        "best_strategy": best_name,
        "best_strategy_sharpe": sr,
        "deflated_sharpe": dsr,
        "dsr_gate": DSR_GATE,
        "passed": bool(pbo < PBO_GATE and dsr > DSR_GATE),
    }
    (args.out / "cpcv_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(is_perf, columns=names).to_csv(args.out / "is_perf.csv", index=False)
    pd.DataFrame(oos_perf, columns=names).to_csv(args.out / "oos_perf.csv", index=False)

    log.info("cpcv.summary", extra=summary)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
