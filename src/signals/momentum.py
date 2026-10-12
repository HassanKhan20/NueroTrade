"""Cross-sectional 12-1 momentum signal — wraps the existing CSM math
into a per-symbol score for the multi-factor ensemble.

This is the same formula used by the live CSM bot, exposed as a clean
function that returns a single number per symbol per date.
"""
from __future__ import annotations

import math

import pandas as pd


def r_12_1(closes: pd.Series, ref_date: pd.Timestamp,
           lookback: int = 252, skip: int = 21) -> float:
    """12-1 momentum: (close at ref-skip) / (close at ref-skip-lookback) - 1.

    Causal: only uses bars strictly before ref_date.
    Returns NaN if insufficient history.
    """
    ts = ref_date.tz_localize(None) if ref_date.tzinfo is not None else ref_date
    prior = closes.loc[closes.index < ts] if isinstance(closes.index, pd.DatetimeIndex) else closes
    if len(prior) < lookback + skip + 1:
        return float("nan")
    end_close = float(prior.iloc[-skip - 1])
    start_close = float(prior.iloc[-skip - lookback - 1])
    if start_close <= 0:
        return float("nan")
    return end_close / start_close - 1.0


def momentum_score(bars: pd.DataFrame, ref_date: pd.Timestamp,
                    lookback: int = 252, skip: int = 21) -> float:
    """Returns the raw 12-1 return for a symbol. Larger = more bullish."""
    if bars is None or bars.empty or "close" not in bars.columns:
        return float("nan")
    df = bars.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return r_12_1(df["close"], ref_date, lookback, skip)
