"""Performance metrics, including Bailey & López de Prado's Deflated Sharpe.

Sharpe is annualized with sqrt(252) for daily returns by default. Inputs are
arithmetic returns unless noted.

References:
  * Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".
  * Bailey & López de Prado (2014), "The Probability of Backtest Overfitting".
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


# --------------------------------------------------------------------------- #
# Basic moments
# --------------------------------------------------------------------------- #
def sharpe_ratio(returns: ArrayLike, *, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0.0:
        return float("nan")
    return float(mu / sd * math.sqrt(periods_per_year))


def max_drawdown(equity_curve: ArrayLike) -> float:
    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())


def profit_factor(pnl_per_trade: ArrayLike) -> float:
    p = np.asarray(pnl_per_trade, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    gains = p[p > 0].sum()
    losses = -p[p < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def win_rate(pnl_per_trade: ArrayLike) -> float:
    p = np.asarray(pnl_per_trade, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    return float((p > 0).mean())


# --------------------------------------------------------------------------- #
# Probabilistic & Deflated Sharpe (Bailey & López de Prado 2014)
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    sr: float,
    n: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """Pr(true SR > benchmark | observed SR), correcting for non-normality.

    sr, sr_benchmark should be in the SAME annualisation as each other (the
    formula is unit-agnostic — but mixing per-period and annualised is wrong).
    n is the number of return observations.
    kurt is the *raw* kurtosis (3 == normal), not excess.
    """
    if n < 2:
        return float("nan")
    excess_kurt = kurt - 3.0
    denom = math.sqrt(
        max(1.0 - skew * sr + (excess_kurt / 4.0) * sr * sr, 1e-12) / (n - 1)
    )
    z = (sr - sr_benchmark) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int) -> float:
    """E[max SR] across N independent trials of N(0,1) Sharpe estimates.

    Uses the Bailey-López de Prado closed-form approximation.
    """
    if n_trials <= 1:
        return 0.0
    eulergamma = 0.5772156649015329
    n = float(n_trials)
    return float(
        (1.0 - eulergamma) * stats.norm.ppf(1.0 - 1.0 / n)
        + eulergamma * stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    )


def deflated_sharpe_ratio(
    sr: float,
    n: int,
    n_trials: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    trial_sr_var: float = 1.0,
) -> float:
    """Probability that the observed Sharpe survives multiple-testing inflation.

    DSR > 0.95 is the standard "deploy" gate.
    """
    if n_trials <= 1:
        return probabilistic_sharpe_ratio(sr, n, skew=skew, kurt=kurt)
    sr_benchmark = math.sqrt(max(trial_sr_var, 0.0)) * expected_max_sharpe(n_trials)
    return probabilistic_sharpe_ratio(
        sr, n, skew=skew, kurt=kurt, sr_benchmark=sr_benchmark
    )


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (PBO)
# --------------------------------------------------------------------------- #
def probability_of_backtest_overfitting(
    is_perf: NDArray[np.float64],
    oos_perf: NDArray[np.float64],
) -> float:
    """Pr(best-IS strategy ranks below median OOS), per López de Prado AFML.

    Inputs are matrices of shape (n_splits, n_strategies) of in-sample and
    out-of-sample performance metrics (e.g. Sharpe). For each split we pick
    the IS-best strategy and check whether its OOS rank is below the median.

    PBO < 0.15 is a common deploy gate.
    """
    is_arr = np.asarray(is_perf, dtype=np.float64)
    oos_arr = np.asarray(oos_perf, dtype=np.float64)
    if is_arr.shape != oos_arr.shape:
        raise ValueError("is_perf and oos_perf must have same shape")
    if is_arr.ndim != 2:
        raise ValueError("is_perf must be 2D (n_splits, n_strategies)")
    n_splits, n_strats = is_arr.shape
    if n_strats < 2:
        return float("nan")
    overfit_count = 0
    for i in range(n_splits):
        best_idx = int(np.argmax(is_arr[i]))
        oos_ranks = stats.rankdata(oos_arr[i], method="average")
        median_rank = (n_strats + 1) / 2.0
        if oos_ranks[best_idx] < median_rank:
            overfit_count += 1
    return overfit_count / n_splits


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    returns: ArrayLike,
    metric_fn,
    *,
    n_boot: int = 5000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Returns (point_estimate, lower, upper) for `metric_fn(returns)`.

    Uses simple stationary IID bootstrap. For autocorrelated returns prefer
    the moving-block bootstrap — TODO once we have real return series.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        nan = float("nan")
        return nan, nan, nan
    rng = rng or np.random.default_rng(0)
    point = float(metric_fn(r))
    samples = np.empty(n_boot, dtype=np.float64)
    n = r.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(r[idx])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return point, lo, hi
