"""Freeze strategy parameters after a CPCV pass.

Reads reports/cpcv/cpcv_summary.json. If `passed: true`, rewrites
config/strategy_params.yaml with `frozen: true`, today's date, and the
current git SHA. Otherwise exits non-zero.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.utils.logging import configure_logging, get_logger

log = get_logger("freeze_params")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    configure_logging("INFO")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cpcv-summary", type=Path, default=Path("reports/cpcv/cpcv_summary.json")
    )
    p.add_argument(
        "--params", type=Path, default=Path("config/strategy_params.yaml")
    )
    args = p.parse_args(argv)

    if not args.cpcv_summary.exists():
        log.error("freeze.no_cpcv_summary", extra={"path": str(args.cpcv_summary)})
        return 1
    summary = json.loads(args.cpcv_summary.read_text())
    if not summary.get("passed"):
        log.error("freeze.cpcv_did_not_pass", extra=summary)
        return 2

    params = yaml.safe_load(args.params.read_text())
    params["frozen"] = True
    params["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    params["git_sha"] = _git_sha()
    params["cpcv_summary"] = {
        "pbo": summary["pbo"],
        "deflated_sharpe": summary["deflated_sharpe"],
        "best_strategy": summary["best_strategy"],
    }
    args.params.write_text(yaml.safe_dump(params, sort_keys=False))
    log.info("freeze.ok", extra={"path": str(args.params)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
