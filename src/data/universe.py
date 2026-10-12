"""Liquid-universe filter.

A name is eligible on date D iff:
  * 20-day average dollar volume (close * volume) >= MIN_DOLLAR_VOL
  * close price on D-1 >= MIN_PRICE
  * symbol is not in the leveraged-ETF blocklist
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

MIN_DOLLAR_VOL = 20_000_000.0
MIN_PRICE = 5.0
DOLLAR_VOL_WINDOW = 20

# Leveraged & inverse ETFs — keep this set conservative; the screen will also
# reject most of these via price/volume but the blocklist is belt-and-braces.
LEVERAGED_ETF_BLOCKLIST: frozenset[str] = frozenset(
    {
        "TQQQ", "SQQQ", "QLD", "PSQ",
        "SPXL", "SPXS", "SSO", "SDS", "UPRO", "SPXU",
        "TNA", "TZA", "URTY", "SRTY",
        "FAS", "FAZ",
        "LABU", "LABD",
        "SOXL", "SOXS",
        "TMF", "TMV",
        "UVXY", "SVXY", "VXX", "VIXY",
        "BOIL", "KOLD", "GUSH", "DRIP",
        "NUGT", "DUST", "JNUG", "JDST",
        "YINN", "YANG",
        "UDOW", "SDOW",
    }
)


@dataclass(frozen=True)
class UniverseFilterParams:
    min_dollar_volume: float = MIN_DOLLAR_VOL
    min_price: float = MIN_PRICE
    window: int = DOLLAR_VOL_WINDOW
    blocklist: frozenset[str] = LEVERAGED_ETF_BLOCKLIST


def is_leveraged_etf(symbol: str, blocklist: frozenset[str] = LEVERAGED_ETF_BLOCKLIST) -> bool:
    return symbol.upper() in blocklist


def liquid_universe(
    daily_bars: Mapping[str, pd.DataFrame],
    as_of: date,
    params: UniverseFilterParams = UniverseFilterParams(),
) -> list[str]:
    """Return symbols eligible to trade on `as_of`.

    Each value in ``daily_bars`` must be a DataFrame indexed by date, with
    'close' and 'volume' columns, sorted ascending.
    """
    eligible: list[str] = []
    for sym, df in daily_bars.items():
        if is_leveraged_etf(sym, params.blocklist):
            continue
        prior = df.loc[df.index < pd.Timestamp(as_of)]
        if len(prior) < params.window:
            continue
        last_close = float(prior["close"].iloc[-1])
        if not np.isfinite(last_close) or last_close < params.min_price:
            continue
        window = prior.iloc[-params.window:]
        avg_dollar_vol = float((window["close"] * window["volume"]).mean())
        if not np.isfinite(avg_dollar_vol) or avg_dollar_vol < params.min_dollar_volume:
            continue
        eligible.append(sym)
    return sorted(eligible)
