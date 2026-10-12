"""CLI: download minute bars for the configured universe and date range.

Usage:
    python -m scripts.download_history --start 2016 --end 2025 \\
        --symbols AAPL MSFT NVDA SPY QQQ

Idempotent: years already on disk are skipped unless --overwrite is passed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config.settings import get_settings
from src.data.ingest_alpaca import AlpacaBarFetcher, ingest_year
from src.utils.logging import configure_logging, get_logger

log = get_logger("download_history")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "ingest.log")

    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True, help="start year, inclusive")
    p.add_argument("--end", type=int, required=True, help="end year, inclusive")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    fetcher = AlpacaBarFetcher(key=settings.alpaca_key, secret=settings.alpaca_secret)

    n_written = n_skipped = n_failed = 0
    for sym in args.symbols:
        for year in range(args.start, args.end + 1):
            try:
                r = ingest_year(
                    fetcher,
                    symbol=sym,
                    year=year,
                    data_dir=settings.data_dir,
                    overwrite=args.overwrite,
                )
                if r.skipped:
                    n_skipped += 1
                else:
                    n_written += 1
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                log.exception("ingest.failed", extra={"symbol": sym, "year": year})
    log.info(
        "download.summary",
        extra={"written": n_written, "skipped": n_skipped, "failed": n_failed},
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
