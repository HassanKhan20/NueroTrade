from __future__ import annotations

import math

import numpy as np
import pytest

from src.utils.indicators import atr, percentile_rank, realized_vol, true_range, zscore


def test_true_range_first_bar_uses_own_close() -> None:
    h = [10.0, 12.0]
    lo = [9.0, 11.0]
    c = [9.5, 11.5]
    tr = true_range(h, lo, c)
    # bar 0: max(10-9, |10-9.5|, |9-9.5|) = 1.0
    assert tr[0] == pytest.approx(1.0)


def test_true_range_uses_prior_close() -> None:
    h = [10.0, 13.0]
    lo = [9.0, 11.0]
    c = [9.5, 12.0]
    tr = true_range(h, lo, c)
    # bar 1: max(13-11, |13-9.5|, |11-9.5|) = 3.5
    assert tr[1] == pytest.approx(3.5)


def test_atr_returns_nan_for_warmup() -> None:
    arr = atr([1, 2, 3], [0, 1, 2], [0.5, 1.5, 2.5], window=14)
    assert np.isnan(arr).all()


def test_atr_simple_average() -> None:
    n = 30
    # Constant close + symmetric high/low keeps true range == 1 every bar.
    c = np.full(n, 100.0)
    h = c + 0.5
    lo = c - 0.5
    a = atr(h, lo, c, window=14)
    assert np.isnan(a[:13]).all()
    assert a[13:] == pytest.approx(1.0)


def test_percentile_rank_constant_history_is_half() -> None:
    assert percentile_rank(5.0, [5.0, 5.0, 5.0]) == pytest.approx(0.5)


def test_percentile_rank_max_value() -> None:
    assert percentile_rank(100.0, [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_percentile_rank_min_value() -> None:
    assert percentile_rank(-100.0, [1, 2, 3, 4, 5]) == pytest.approx(0.0)


def test_zscore_standard_normal_like() -> None:
    np.random.seed(0)
    h = np.random.normal(loc=0.0, scale=1.0, size=1000)
    z = zscore(2.0, h)
    assert 1.5 < z < 2.5


def test_zscore_constant_history_zero() -> None:
    assert zscore(5.0, [3.0, 3.0, 3.0]) == 0.0


def test_realized_vol_warmup_is_nan() -> None:
    rv = realized_vol([0.01, 0.02, -0.01], window=14)
    assert np.isnan(rv).all()


def test_realized_vol_matches_numpy_std() -> None:
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0, 0.01, size=100)
    rv = realized_vol(rets, window=20)
    assert rv[19] == pytest.approx(rets[:20].std(ddof=1))
    assert rv[99] == pytest.approx(rets[80:100].std(ddof=1))
