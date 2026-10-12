from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.universe import (
    LEVERAGED_ETF_BLOCKLIST,
    UniverseFilterParams,
    is_leveraged_etf,
    liquid_universe,
)


def _daily(symbol: str, *, price: float, volume: float, days: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2025-04-01", periods=days)
    return pd.DataFrame(
        {"close": [price] * days, "volume": [volume] * days},
        index=idx,
    )


def test_blocklist_contains_known_3x_etfs() -> None:
    for sym in ("TQQQ", "SQQQ", "SPXL", "TNA", "SOXL", "UVXY"):
        assert sym in LEVERAGED_ETF_BLOCKLIST
        assert is_leveraged_etf(sym)


def test_passes_liquid_name() -> None:
    bars = {"AAPL": _daily("AAPL", price=200.0, volume=500_000)}
    out = liquid_universe(bars, as_of=date(2025, 4, 30))
    assert out == ["AAPL"]


def test_rejects_low_price() -> None:
    bars = {"PENNY": _daily("PENNY", price=2.0, volume=10_000_000)}
    out = liquid_universe(bars, as_of=date(2025, 4, 30))
    assert out == []


def test_rejects_low_dollar_volume() -> None:
    # 50 * 100k = $5M average, below default $20M floor
    bars = {"SMALL": _daily("SMALL", price=50.0, volume=100_000)}
    out = liquid_universe(bars, as_of=date(2025, 4, 30))
    assert out == []


def test_rejects_leveraged_etfs() -> None:
    bars = {"TQQQ": _daily("TQQQ", price=70.0, volume=10_000_000)}
    out = liquid_universe(bars, as_of=date(2025, 4, 30))
    assert out == []


def test_requires_minimum_history() -> None:
    bars = {"FRESH": _daily("FRESH", price=50.0, volume=1_000_000, days=10)}
    out = liquid_universe(bars, as_of=date(2025, 4, 30))
    assert out == []


def test_returns_sorted() -> None:
    bars = {
        "ZZZ": _daily("ZZZ", price=100.0, volume=500_000),
        "AAA": _daily("AAA", price=100.0, volume=500_000),
        "MMM": _daily("MMM", price=100.0, volume=500_000),
    }
    out = liquid_universe(bars, as_of=date(2025, 4, 30))
    assert out == ["AAA", "MMM", "ZZZ"]
