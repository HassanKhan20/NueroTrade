"""Component C — Volatility regime overlay.

Returns a multiplier in {0.0, 0.5, 1.0} that the allocator applies on top of
the per-component risk budget:

  * VIX percentile rank > halt_at_vix_pct           -> 0.0  (do not trade)
  * VIX percentile rank > reduce_at_vix_pct OR
    realized-vol z-score > reduce_at_vol_z          -> 0.5  (half size)
  * otherwise                                       -> 1.0  (full size)

Intentionally cautious: this is a survival rule, not an alpha source. It
removes meaningful drawdown at small Sharpe cost in published trend/momentum
backtests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.indicators import percentile_rank, realized_vol, zscore


@dataclass(frozen=True)
class VolRegimeParams:
    vix_lookback_days: int = 252
    realized_vol_window: int = 20
    realized_vol_history_years: int = 5
    reduce_at_vix_pct: float = 0.90
    reduce_at_vol_z: float = 2.0
    halt_at_vix_pct: float = 0.97


def regime_multiplier(
    vix_today: float,
    vix_history: pd.Series,
    spy_close_history: pd.Series,
    params: VolRegimeParams = VolRegimeParams(),
) -> float:
    """Compute the regime multiplier for the next session.

    Args:
        vix_today: latest VIX close (e.g. yesterday's close, used to size today).
        vix_history: tailing VIX closes (will be sliced to lookback_days).
        spy_close_history: trailing daily SPY closes (>= ~5y for the z-score).
    """
    if not np.isfinite(vix_today):
        return 0.0  # data anomaly => sit out

    vh = vix_history.dropna().to_numpy(dtype=float)
    if vh.size < params.vix_lookback_days:
        return 1.0  # not enough data to gate; allow trading
    vix_pct = percentile_rank(vix_today, vh[-params.vix_lookback_days :])

    if vix_pct > params.halt_at_vix_pct:
        return 0.0

    spy = spy_close_history.dropna().to_numpy(dtype=float)
    needed = params.realized_vol_history_years * 252 + params.realized_vol_window
    if spy.size < needed:
        if vix_pct > params.reduce_at_vix_pct:
            return 0.5
        return 1.0

    rets = np.diff(np.log(spy))
    rv = realized_vol(rets, window=params.realized_vol_window)
    rv = rv[~np.isnan(rv)]
    rv_history = rv[-(params.realized_vol_history_years * 252) :]
    z = zscore(float(rv[-1]), rv_history)

    if vix_pct > params.reduce_at_vix_pct or z > params.reduce_at_vol_z:
        return 0.5
    return 1.0
