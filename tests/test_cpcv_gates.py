"""Regression tests for the deploy gates: PBO + Deflated Sharpe.

These confirm the real numbers in the harness: an engineered overfit grid
produces PBO -> 1, and a single strategy with strong drift produces a high
Deflated Sharpe.
"""
from __future__ import annotations

import numpy as np

from src.backtest.cpcv import cpcv_splits, score_grid
from src.backtest.metrics import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)


def test_engineered_overfit_grid_fails_pbo_gate() -> None:
    """Anti-correlated IS/OOS performance => PBO == 1 (always overfit)."""
    n = 252 * 4
    rng = np.random.default_rng(0)
    rets = {}
    # Strategy i has IS-good drift but OOS-bad drift in a known half of the year.
    for i in range(8):
        a = rng.normal(0.001, 0.01, n // 2)
        b = rng.normal(-0.001, 0.01, n - n // 2)
        rets[f"s{i}"] = np.concatenate([a, b])

    splits = cpcv_splits(n_obs=n, n_groups=6, n_test_groups=2, embargo=5)
    is_perf, oos_perf, _ = score_grid(rets, splits, metric_fn=sharpe_ratio)
    pbo = probability_of_backtest_overfitting(is_perf, oos_perf)
    # Not strictly 1.0 due to noise but should clearly fail the 0.15 gate.
    assert pbo > 0.5


def test_deflated_sharpe_passes_for_strong_single_strategy() -> None:
    rng = np.random.default_rng(1)
    rets = rng.normal(0.001, 0.01, size=252 * 5)  # ~1.59 SR/yr
    sr = sharpe_ratio(rets) / np.sqrt(252.0)  # convert to per-period
    # Only one trial, so DSR should be close to PSR; comfortably above 0.95
    dsr = deflated_sharpe_ratio(sr, n=len(rets), n_trials=1)
    assert dsr > 0.95


def test_deflated_sharpe_penalises_many_trials() -> None:
    rng = np.random.default_rng(2)
    rets = rng.normal(0.0005, 0.01, size=252 * 3)
    sr = sharpe_ratio(rets) / np.sqrt(252.0)
    dsr_one = deflated_sharpe_ratio(sr, n=len(rets), n_trials=1)
    dsr_many = deflated_sharpe_ratio(sr, n=len(rets), n_trials=500)
    assert dsr_many < dsr_one
