"""Sanity-check Alpaca credentials and data access.

Run this once after editing .env. Exits 0 if everything works, non-zero if
keys are missing/wrong, account isn't reachable, or the data API is down.

  python -m scripts.check_connection
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config.settings import get_settings
from src.data.ingest_alpaca import AlpacaBarFetcher
from src.live.alpaca_client import AlpacaBrokerClient
from src.utils.logging import configure_logging, get_logger

log = get_logger("check_connection")


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    if not settings.alpaca_key or not settings.alpaca_secret:
        log.error("check.no_credentials")
        return 1
    if not settings.alpaca_paper:
        log.error("check.live_mode_blocked",
                  extra={"hint": "set ALPACA_PAPER=true while validating"})
        return 1

    log.info(
        "check.credentials_ok",
        extra={
            "key_prefix": settings.alpaca_key[:4] + "...",
            "paper": settings.alpaca_paper,
        },
    )

    # 1) Account
    try:
        broker = AlpacaBrokerClient(
            key=settings.alpaca_key,
            secret=settings.alpaca_secret,
            paper=settings.alpaca_paper,
        )
        acct = broker.account()
        log.info(
            "check.account_ok",
            extra={
                "equity": acct.equity,
                "cash": acct.cash,
                "buying_power": acct.buying_power,
            },
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("check.account_failed")
        return 2

    # 2) Bar data fetch — use a window 30 days old so the IEX-only free tier
    #    can serve it (free tier blocks anything < ~15 min old as "SIP data").
    try:
        end = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=30)
        start = end - timedelta(days=2)
        fetcher = AlpacaBarFetcher(settings.alpaca_key, settings.alpaca_secret)
        bars = fetcher.fetch("AAPL", start, end)
        log.info("check.data_ok", extra={"rows": int(len(bars))})
    except Exception as exc:  # noqa: BLE001
        log.exception("check.data_failed")
        return 3

    log.info("check.all_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
