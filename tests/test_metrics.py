from __future__ import annotations

import math

import numpy as np
import pytest

from src.backtest.metrics import (
    bootstrap_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    profit_factor,
    sharpe_ratio,
    win_rate,
)


def test_sharpe_zero_for_zero_returns() -> None:
    assert math.isnan(sharpe_ratio([0.0, 0.0, 0.0, 0.0]))


def test_sharpe_positive_for_drift_up() -> None:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=2520)
    sr = sharpe_ratio(rets)
    assert 0.5 < sr < 3.0


def test_max_drawdown_simple() -> None:
    eq = [100.0, 110.0, 99.0, 90.0, 120.0]
    # peak 110 -> trough 90 -> dd = -20/110
    assert max_drawdown(eq) == pytest.approx(-20.0 / 110.0)


def test_profit_factor_basic() -> None:
    pnl = [10.0, -5.0, 20.0, -10.0]  # gains 30, losses 15
    assert profit_factor(pnl) == pytest.approx(2.0)


def test_profit_factor_no_losses() -> None:
    assert profit_factor([1.0, 2.0, 3.0]) == math.inf


def test_win_rate() -> None:
    assert win_rate([1.0, -1.0, 1.0, 1.0]) == pytest.approx(0.75)


def test_psr_at_observed_value_is_half_for_normal() -> None:
    # With benchmark equal to observed SR, PSR == 0.5 exactly under normality.
    psr = probabilistic_sharpe_ratio(sr=1.0, n=252, sr_benchmark=1.0)
    assert psr == pytest.approx(0.5, abs=1e-6)


def test_psr_increases_with_sample_size() -> None:
    # SR is per-period; annualized SR=1 corresponds to ~0.063 daily.
    sr_per_period = 0.063
    a = probabilistic_sharpe_ratio(sr=sr_per_period, n=252, sr_benchmark=0.0)
    b = probabilistic_sharpe_ratio(sr=sr_per_period, n=2520, sr_benchmark=0.0)
    assert b > a


def test_expected_max_sharpe_grows_with_n_trials() -> None:
    assert expected_max_sharpe(2) < expected_max_sharpe(20) < expected_max_sharpe(200)


def test_deflated_sharpe_penalises_many_trials() -> None:
    one = deflated_sharpe_ratio(sr=1.0, n=252, n_trials=1)
    many = deflated_sharpe_ratio(sr=1.0, n=252, n_trials=200)
    assert one > many


def test_pbo_overfit_synthetic() -> None:
    """Best IS strategy is engineered to be the worst OOS -> high PBO."""
    n_splits, n_strats = 6, 8
    rng = np.random.default_rng(0)
    is_perf = rng.normal(size=(n_splits, n_strats))
    oos_perf = -is_perf  # exact anti-correlation -> max PBO
    assert probability_of_backtest_overfitting(is_perf, oos_perf) == pytest.approx(1.0)


def test_pbo_zero_when_perfect_carryover() -> None:
    n_splits, n_strats = 6, 8
    rng = np.random.default_rng(1)
    is_perf = rng.normal(size=(n_splits, n_strats))
    oos_perf = is_perf.copy()  # perfect carryover -> PBO = 0
    assert probability_of_backtest_overfitting(is_perf, oos_perf) == 0.0


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(2)
    rets = rng.normal(0.001, 0.01, size=500)
    point, lo, hi = bootstrap_ci(rets, np.mean, n_boot=500, rng=rng)
    assert lo <= point <= hi
