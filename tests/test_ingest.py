from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingest_alpaca import (
    REQUIRED_COLUMNS,
    BarFetcher,
    _validate_frame,
    ingest_year,
    load_range,
    load_year,
)


class FakeFetcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, datetime, datetime]] = []

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        idx = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")[:5]
        df = pd.DataFrame(
            {
                "open": [100.0] * len(idx),
                "high": [101.0] * len(idx),
                "low": [99.0] * len(idx),
                "close": [100.5] * len(idx),
                "volume": [1000.0] * len(idx),
            },
            index=idx,
        )
        return df


def test_validate_rejects_missing_columns() -> None:
    df = pd.DataFrame({"open": [1.0]}, index=pd.DatetimeIndex(["2025-01-01"], tz="UTC"))
    with pytest.raises(ValueError):
        _validate_frame(df)


def test_validate_rejects_naive_index_naively_fixed() -> None:
    idx = pd.DatetimeIndex(["2025-01-01 09:30"])  # naive
    df = pd.DataFrame(
        {c: [1.0] for c in REQUIRED_COLUMNS}, index=idx
    )
    out = _validate_frame(df)
    assert out.index.tz is not None


def test_ingest_year_writes_and_skips(tmp_path: Path) -> None:
    fetcher = FakeFetcher()
    r1 = ingest_year(fetcher, symbol="AAPL", year=2024, data_dir=tmp_path)
    assert r1.skipped is False
    assert r1.rows_written > 0
    # idempotency: second run is a no-op
    r2 = ingest_year(fetcher, symbol="AAPL", year=2024, data_dir=tmp_path)
    assert r2.skipped is True
    assert len(fetcher.calls) == 1


def test_load_year_round_trip(tmp_path: Path) -> None:
    ingest_year(FakeFetcher(), symbol="AAPL", year=2024, data_dir=tmp_path)
    df = load_year("AAPL", 2024, tmp_path)
    assert not df.empty
    assert list(df.columns) == list(REQUIRED_COLUMNS)


def test_load_range_concatenates_years(tmp_path: Path) -> None:
    ingest_year(FakeFetcher(), symbol="AAPL", year=2023, data_dir=tmp_path)
    ingest_year(FakeFetcher(), symbol="AAPL", year=2024, data_dir=tmp_path)
    df = load_range("AAPL", start=__import__("datetime").date(2023, 1, 1),
                    end=__import__("datetime").date(2024, 12, 31),
                    data_dir=tmp_path)
    # Each fetch returns 5 rows starting at year boundary
    assert len(df) >= 5
