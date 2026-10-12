"""Bulk-fetch DAILY bars for a universe and store under data/parquet_daily/.

Daily bars are tiny (~252 rows/year) so we do one file per symbol covering
the whole window. Idempotent — re-running with --append updates the cache
with any new days at the tail.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.data.ingest_alpaca import REQUIRED_COLUMNS, AlpacaBarFetcher
from src.utils.logging import configure_logging, get_logger

log = get_logger("download_daily")


def _path_for(root: Path, symbol: str) -> Path:
    return root / "parquet_daily" / f"{symbol.upper()}.parquet"


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir / "download_daily.log")

    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--symbols", nargs="+", required=False)
    p.add_argument(
        "--universe", default=None,
        choices=[None, "all", "sp100", "halal", "midcap", "biotech"],
        help="use a curated universe (all/sp100/halal/midcap/biotech)",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    syms: list[str]
    if args.universe == "all":
        from config.universe_meta import all_symbols
        syms = list(all_symbols())
    elif args.universe == "sp100":
        from config.sp100_universe import SP100_UNIVERSE
        syms = list(SP100_UNIVERSE)
    elif args.universe == "halal":
        from config.halal_universe import HALAL_UNIVERSE
        syms = list(HALAL_UNIVERSE)
    elif args.universe == "midcap":
        from config.midcap_universe import MIDCAP_UNIVERSE
        syms = list(MIDCAP_UNIVERSE)
    elif args.universe == "biotech":
        from config.biotech_universe import BIOTECH_UNIVERSE
        syms = list(BIOTECH_UNIVERSE)
    elif args.symbols:
        syms = list(args.symbols)
    else:
        log.error("download_daily.no_symbols")
        return 1

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    # Clamp end to safely-historical (Alpaca free tier rejects recent SIP).
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    safe_now = datetime.now(timezone.utc) - timedelta(hours=24)
    if end > safe_now:
        end = safe_now

    fetcher = AlpacaBarFetcher(settings.alpaca_key, settings.alpaca_secret)
    n_written = n_skipped = n_failed = 0
    for sym in syms:
        out = _path_for(settings.data_dir, sym)
        if out.exists() and not args.overwrite:
            log.info("download_daily.skip", extra={"symbol": sym, "path": str(out)})
            n_skipped += 1
            continue
        try:
            df = fetcher.fetch_daily(sym, start, end)
            if df.empty:
                log.warning("download_daily.empty", extra={"symbol": sym})
                n_failed += 1
                continue
            if df.index.tz is None:
                df = df.tz_localize("UTC")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df = df[list(REQUIRED_COLUMNS)]
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=True)
            log.info(
                "download_daily.write",
                extra={"symbol": sym, "rows": int(len(df)), "path": str(out)},
            )
            n_written += 1
        except Exception:  # noqa: BLE001
            log.exception("download_daily.failed", extra={"symbol": sym})
            n_failed += 1
    log.info(
        "download_daily.summary",
        extra={"written": n_written, "skipped": n_skipped, "failed": n_failed},
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
