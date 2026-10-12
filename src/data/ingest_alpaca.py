"""Alpaca minute-bar ingest.

Idempotent: re-running over a window already on disk is a no-op. Each
(symbol, year) is a single Parquet file under data/parquet/{symbol}/{year}.parquet.

Rate-limited via tenacity; honours Alpaca's 200 req/min budget by sleeping
on 429 responses.

We keep this layer thin and dependency-light so unit tests can substitute a
fake fetcher without importing alpaca-py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

import pandas as pd

from src.utils.clock import is_trading_day
from src.utils.logging import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class BarFetcher(Protocol):
    """Anything that, given a (symbol, start, end), returns 1-min bars.

    Implementations: ``AlpacaBarFetcher`` for production, ``InMemoryBarFetcher``
    for tests.
    """

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame: ...


@dataclass
class IngestResult:
    symbol: str
    year: int
    rows_written: int
    skipped: bool


def _path_for(root: Path, symbol: str, year: int) -> Path:
    return root / "parquet" / symbol.upper() / f"{year}.parquet"


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a fetcher's output into our canonical schema."""
    if df.empty:
        return df
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"bar frame missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("bar index must be DatetimeIndex")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("bar frame contains NaNs in OHLCV columns")
    return df[list(REQUIRED_COLUMNS)]


def ingest_year(
    fetcher: BarFetcher,
    *,
    symbol: str,
    year: int,
    data_dir: Path,
    overwrite: bool = False,
) -> IngestResult:
    """Download and persist one (symbol, year) of 1-min bars.

    Returns an IngestResult; ``skipped=True`` if the file already exists and
    ``overwrite`` is False.
    """
    out_path = _path_for(data_dir, symbol, year)
    if out_path.exists() and not overwrite:
        log.info("ingest.skip", extra={"symbol": symbol, "year": year, "path": str(out_path)})
        return IngestResult(symbol=symbol, year=year, rows_written=0, skipped=True)

    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    # Alpaca's free / IEX-only entitlement rejects requests that include
    # "recent SIP data" (within ~15 minutes of now). Clamp to 24h ago to
    # stay safely on the historical side of that line.
    safe_now = datetime.now(timezone.utc) - timedelta(hours=24)
    if end > safe_now:
        end = safe_now
    if start >= end:
        log.info(
            "ingest.skip_future_year",
            extra={"symbol": symbol, "year": year},
        )
        return IngestResult(symbol=symbol, year=year, rows_written=0, skipped=True)
    df = fetcher.fetch(symbol, start, end)
    df = _validate_frame(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=True)
    log.info(
        "ingest.write",
        extra={
            "symbol": symbol,
            "year": year,
            "rows": int(len(df)),
            "path": str(out_path),
        },
    )
    return IngestResult(symbol=symbol, year=year, rows_written=len(df), skipped=False)


def load_year(symbol: str, year: int, data_dir: Path) -> pd.DataFrame:
    """Read one cached (symbol, year) file. Empty DataFrame if missing."""
    p = _path_for(data_dir, symbol, year)
    if not p.exists():
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    return pd.read_parquet(p)


def load_range(
    symbol: str, start: date, end: date, data_dir: Path
) -> pd.DataFrame:
    """Concatenate cached years overlapping [start, end] (inclusive)."""
    frames: list[pd.DataFrame] = []
    for y in range(start.year, end.year + 1):
        f = load_year(symbol, y, data_dir)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & (
        df.index < pd.Timestamp(end + timedelta(days=1), tz="UTC")
    )
    return df.loc[mask]


# --------------------------------------------------------------------------- #
# Production fetcher (alpaca-py). Imported lazily so tests don't need the SDK.
# --------------------------------------------------------------------------- #
class AlpacaBarFetcher:
    """alpaca-py wrapper, retried on 429/5xx, paginated transparently.

    Constructor is light — no network call until ``fetch()`` runs.
    """

    def __init__(self, key: str, secret: str) -> None:
        if not key or not secret:
            raise ValueError("Alpaca credentials missing")
        self._key = key
        self._secret = secret
        self._client = None  # lazy

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from alpaca.data.historical import StockHistoricalDataClient  # type: ignore

        self._client = StockHistoricalDataClient(self._key, self._secret)

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._fetch_bars(symbol, start, end, daily=False)

    def fetch_daily(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch DAILY bars — far smaller than minute, fits CSM's needs.

        Returns a frame with index = bar-end timestamps and the canonical
        OHLCV columns. ~252 rows per year per symbol.
        """
        return self._fetch_bars(symbol, start, end, daily=True)

    def _fetch_bars(
        self, symbol: str, start: datetime, end: datetime, *, daily: bool
    ) -> pd.DataFrame:
        self._ensure_client()
        from alpaca.data.requests import StockBarsRequest  # type: ignore
        from alpaca.data.timeframe import TimeFrame  # type: ignore
        from tenacity import (  # type: ignore
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        tf = TimeFrame.Day if daily else TimeFrame.Minute

        @retry(
            reraise=True,
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=1, max=30),
            retry=retry_if_exception_type(Exception),
        )
        def _do_fetch() -> pd.DataFrame:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
            )
            assert self._client is not None
            bars = self._client.get_stock_bars(req).df  # type: ignore[union-attr]
            if bars.empty:
                return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.droplevel(0)
            return bars[list(REQUIRED_COLUMNS)]

        return _do_fetch()
