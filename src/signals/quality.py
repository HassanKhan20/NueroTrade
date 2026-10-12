"""Quality factor — STUB implementation pending fundamentals data feed.

Quality is normally computed from accounting data (ROE, debt/equity,
earnings stability, gross margin trend). Without a fundamentals API
wired in, we offer two paths:

  1. **Hand-curated quality scores** in QUALITY_OVERRIDES — for the
     ~150 names in the halal universe, drawn from public ETF holdings
     of ETFs that already weight by quality (QUAL, MTUM-Q overlap).

  2. **Price-derived proxy**: the long-run *stability* of returns is a
     valid quality proxy. Stocks with smooth equity curves
     (low coefficient-of-variation of monthly returns) tend to be
     higher-quality businesses.

Both are conservative. Replace with an EDGAR / FactSet / SimFin pull
when a feed is available — the call signature is `quality_score(symbol,
bars, ref_date)` and any feed can be wired behind it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Hand-tagged quality scores in {-1, 0, +1} for known halal-universe names.
# +1 = consistent earnings + low debt + durable franchise (per QUAL ETF
#      heuristics). 0 = average. -1 = cyclical / capital-intensive / debt-heavy.
QUALITY_OVERRIDES: dict[str, int] = {
    # Mega-cap quality leaders (high ROE, low debt, stable margins)
    "AAPL": 1, "MSFT": 1, "GOOGL": 1, "GOOG": 1, "META": 1,
    "NVDA": 1, "AVGO": 1, "ORCL": 1, "ADBE": 1, "INTU": 1,
    "ACN": 1, "TXN": 1, "ASML": 1,
    # Healthcare quality leaders
    "JNJ": 1, "LLY": 1, "ABT": 1, "TMO": 1, "DHR": 1,
    "MRK": 1, "AMGN": 1, "ISRG": 1, "DXCM": 1,
    # Consumer staples / quality compounders
    "PG": 1, "KO": 1, "PEP": 1, "COST": 1, "WMT": 1, "MDLZ": 1,
    # Industrials (selectively)
    "CAT": 1, "DE": 1, "HON": 1, "ETN": 1, "ROK": 1, "ITW": 1,
    "UNP": 1, "FAST": 1,
    # Materials, energy: cyclical → 0
    "FCX": 0, "NEM": 0, "XOM": 0, "CVX": 0, "COP": 0, "EOG": 0,
    # Mid-quality / cyclical tech
    "AMD": 0, "INTC": 0, "MU": 0, "QCOM": 0, "AMAT": 0, "LRCX": 0,
    # REITs / utilities (capital-intensive)
    "AMT": 0, "PLD": 0, "EQIX": 0, "NEE": 0, "DUK": 0,
    # Smaller / less proven
    "CRM": 0, "NOW": 1, "PANW": 0, "CRWD": 0, "FTNT": 0,
}


def stability_score(closes: pd.Series, lookback_days: int = 252) -> float:
    """Return -coefficient-of-variation of last-year monthly returns.
    Smoother price action → smaller |CV| → larger score."""
    if len(closes) < lookback_days + 21:
        return float("nan")
    monthly = closes.iloc[-lookback_days:].resample("ME").last().pct_change().dropna()
    if len(monthly) < 6:
        return float("nan")
    mu = float(monthly.mean())
    sd = float(monthly.std(ddof=1))
    if sd <= 0:
        return float("nan")
    cv = sd / max(abs(mu), 1e-6)
    # Map cv to a [-2, +2] score: low cv good, high cv bad.
    return float(np.clip(2.0 - cv, -2.0, 2.0))


def quality_score(symbol: str, bars: pd.DataFrame | None,
                   ref_date: pd.Timestamp) -> float:
    """Return a quality score in approximately [-2, +2].

    Order of preference:
      1. Hand-curated override
      2. Price-stability proxy
      3. NaN (caller should treat as neutral)
    """
    if symbol.upper() in QUALITY_OVERRIDES:
        return float(QUALITY_OVERRIDES[symbol.upper()])

    if bars is None or bars.empty or "close" not in bars.columns:
        return float("nan")
    df = bars.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    ts = ref_date.tz_localize(None) if ref_date.tzinfo is not None else ref_date
    prior = df.loc[df.index < ts]
    return stability_score(prior["close"])
