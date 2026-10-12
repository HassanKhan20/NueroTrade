"""Synthetic-data builders shared across end-to-end tests.

Produces deterministic 1-min OHLCV frames in NY-tz for any number of trading
days. Used by the engine-level and CPCV-level integration tests so we can
exercise the full pipeline without needing real market data on disk.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.clock import NY_TZ, is_trading_day, trading_days


def synth_minute_day(
    d: date,
    *,
    open_price: float,
    drift_per_min: float = 0.0,
    range_pct: float = 0.001,
    or_volume: float = 5_000_000.0,
    rest_volume: float = 500_000.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """One full RTH session (09:30-15:59) of 1-min bars."""
    rng = np.random.default_rng(seed if seed is not None else d.toordinal())
    rows = []
    cur = datetime.combine(d, time(9, 30), tzinfo=NY_TZ)
    end = datetime.combine(d, time(16, 0), tzinfo=NY_TZ)
    price = open_price
    while cur < end:
        noise = rng.normal(0.0, max(price * range_pct * 0.5, 1e-6))
        c = price + noise
        h = max(price, c) + abs(rng.normal(0.0, price * range_pct))
        lo = min(price, c) - abs(rng.normal(0.0, price * range_pct))
        v = or_volume if cur.time() < time(9, 35) else rest_volume
        rows.append((cur, price, h, lo, c, v))
        price = c + drift_per_min
        cur += timedelta(minutes=1)
    return pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    ).set_index("ts")


def synth_minute_history(
    symbol: str,
    days: list[date],
    *,
    open_price: float,
    drift_per_min: float = 0.0,
    range_pct: float = 0.001,
    or_volume: float = 5_000_000.0,
    rest_volume: float = 500_000.0,
) -> pd.DataFrame:
    """Concatenate one day's frames for a list of dates into one frame.

    The closing price of one day becomes the next day's open (no overnight gap).
    """
    frames = []
    last_close = open_price
    for d in days:
        if not is_trading_day(d):
            continue
        f = synth_minute_day(
            d,
            open_price=last_close,
            drift_per_min=drift_per_min,
            range_pct=range_pct,
            or_volume=or_volume,
            rest_volume=rest_volume,
            seed=hash((symbol, d.toordinal())) & 0xFFFFFFFF,
        )
        frames.append(f)
        last_close = float(f["close"].iloc[-1])
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(frames).sort_index()


def daily_from_minute(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min bars to a daily OHLCV index aligned to the bar's date."""
    if minute_bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    agg = minute_bars.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return agg
