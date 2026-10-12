"""Slippage model.

We assume each fill pays half-spread + a small linear impact term. Spread is
estimated from the bar's high-low range when explicit bid/ask is unavailable
(typical for Alpaca free-tier IEX bars).

Conventions:
* `side` is +1 for buy, -1 for sell.
* `bar` is a dict-like with at least 'high' and 'low'.
* Output is the *fill price*, never the slippage delta — callers don't need
  to remember to add or subtract.
"""
from __future__ import annotations

from typing import Mapping

DEFAULT_IMPACT_BPS = 0.5  # 0.5 bp linear impact
DEFAULT_HALF_SPREAD_FRAC_OF_RANGE = 0.25  # half-spread ≈ 25% of bar range


def fill_price(
    side: int,
    ref_price: float,
    bar: Mapping[str, float],
    *,
    impact_bps: float = DEFAULT_IMPACT_BPS,
    half_spread_frac: float = DEFAULT_HALF_SPREAD_FRAC_OF_RANGE,
) -> float:
    """Return the simulated fill price.

    A buy fills above ref_price, a sell fills below — by half-spread + impact.
    """
    if side not in (-1, 1):
        raise ValueError(f"side must be +1 or -1, got {side}")
    rng = max(bar["high"] - bar["low"], 0.0)
    half_spread = rng * half_spread_frac * 0.5
    impact = ref_price * (impact_bps / 10_000.0)
    delta = half_spread + impact
    return ref_price + side * delta


def slippage_amount(
    side: int,
    ref_price: float,
    bar: Mapping[str, float],
    *,
    impact_bps: float = DEFAULT_IMPACT_BPS,
    half_spread_frac: float = DEFAULT_HALF_SPREAD_FRAC_OF_RANGE,
) -> float:
    """Convenience: cost-of-slippage in price units (always >= 0)."""
    fp = fill_price(
        side,
        ref_price,
        bar,
        impact_bps=impact_bps,
        half_spread_frac=half_spread_frac,
    )
    return abs(fp - ref_price)
