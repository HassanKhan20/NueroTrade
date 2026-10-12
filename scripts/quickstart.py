"""End-to-end orchestrator: download history, run IS backtest, build grid,
run CPCV, freeze params, run holdout. Skips any phase whose artefact already
exists, so it's safe to re-run after a partial completion.

This is a convenience wrapper — every stage can also be invoked individually.
The CPCV gate and holdout gate exit non-zero on failure; this script
preserves that behaviour.

  python -m scripts.quickstart --start 2016 --end 2025 --symbols AAPL ...
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from config.settings import get_settings
from src.utils.logging import configure_logging, get_logger

log = get_logger("quickstart")


def _run(cmd: list[str]) -> int:
    log.info("quickstart.exec", extra={"cmd": shlex.join(cmd)})
    rc = subprocess.call([sys.executable, *cmd])
    log.info("quickstart.exec_done", extra={"cmd": cmd[0], "rc": rc})
    return rc


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "quickstart.log")

    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2016, help="start year (inclusive)")
    p.add_argument("--end", type=int, default=2025, help="end year (inclusive)")
    p.add_argument("--is-end", default="2023-12-31",
                   help="last date of in-sample window (YYYY-MM-DD)")
    p.add_argument("--oos-start", default="2024-01-01")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--momentum-symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--vix-symbol", default="VIXY")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args(argv)

    is_start = f"{args.start}-01-01"
    is_end = args.is_end
    oos_start = args.oos_start
    oos_end = f"{args.end}-12-31"

    # 1) Download history (idempotent)
    if not args.skip_download:
        rc = _run(
            [
                "-m", "scripts.download_history",
                "--start", str(args.start),
                "--end", str(args.end),
                "--symbols", *args.symbols, *args.momentum_symbols, args.vix_symbol,
            ]
        )
        if rc != 0:
            log.error("quickstart.download_failed", extra={"rc": rc})
            return rc

    # 2) IS backtest
    rc = _run(
        [
            "-m", "scripts.run_backtest",
            "--start", is_start, "--end", is_end,
            "--symbols", *args.symbols,
            "--momentum-symbols", *args.momentum_symbols,
            "--vix-symbol", args.vix_symbol,
            "--label", "is",
        ]
    )
    if rc != 0:
        return rc

    # 3) Build grid for CPCV
    rc = _run(
        [
            "-m", "scripts.build_grid",
            "--start", is_start, "--end", is_end,
            "--symbols", *args.symbols,
            "--momentum-symbols", *args.momentum_symbols,
            "--vix-symbol", args.vix_symbol,
        ]
    )
    if rc != 0:
        return rc

    # 4) CPCV (hard gate)
    rc = _run(["-m", "scripts.run_cpcv",
               "--returns", "reports/grid_returns.parquet"])
    if rc != 0:
        log.error("quickstart.cpcv_gate_failed", extra={"rc": rc})
        return rc

    # 5) Freeze params (only proceeds if CPCV passed)
    rc = _run(["-m", "scripts.freeze_params"])
    if rc != 0:
        return rc

    # 6) OOS holdout backtest
    rc = _run(
        [
            "-m", "scripts.run_backtest",
            "--start", oos_start, "--end", oos_end,
            "--symbols", *args.symbols,
            "--momentum-symbols", *args.momentum_symbols,
            "--vix-symbol", args.vix_symbol,
            "--label", "oos",
        ]
    )
    if rc != 0:
        return rc

    # 7) Locate the most recent IS + OOS report dirs and run holdout gate
    reports = Path("reports/backtests")
    is_dirs = sorted(reports.glob("is_*"), key=lambda p: p.stat().st_mtime)
    oos_dirs = sorted(reports.glob("oos_*"), key=lambda p: p.stat().st_mtime)
    if not is_dirs or not oos_dirs:
        log.error("quickstart.report_dirs_missing")
        return 5
    is_dir = is_dirs[-1]
    oos_dir = oos_dirs[-1]
    rc = _run(
        [
            "-m", "scripts.run_holdout",
            "--is-summary", str(is_dir / "summary.json"),
            "--oos-summary", str(oos_dir / "summary.json"),
            "--oos-equity", str(oos_dir / "equity_curve.csv"),
        ]
    )
    if rc != 0:
        log.error("quickstart.holdout_gate_failed", extra={"rc": rc})
        return rc

    log.info("quickstart.all_gates_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
