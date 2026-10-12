"""Combinatorial Purged Cross-Validation (López de Prado, AFML Ch. 7).

We split a sequence of T trading days into N groups, then for each
combination of K test groups:
  * train (= "in-sample") on the remaining N-K groups,
  * test (= "out-of-sample") on the K groups,
with an embargo of E days at each train/test boundary to guard against
information leakage from autocorrelated returns.

For our purposes "training" means scoring a strategy variant on the IS data
and "testing" means scoring the same variant on the OOS data — we do not
fit parameters here; the harness exists to estimate how IS-best variants
generalise.

Returns a list of (is_indices, oos_indices) splits.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


def cpcv_splits(
    n_obs: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 0,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Yield (in_sample_idx, out_of_sample_idx) for each combinatorial split.

    `embargo` purges that many observations around each test group.
    """
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be in [1, n_groups-1]")
    if n_obs < n_groups:
        raise ValueError(f"need at least {n_groups} observations")

    boundaries = np.linspace(0, n_obs, n_groups + 1, dtype=int)
    groups = [
        np.arange(boundaries[i], boundaries[i + 1], dtype=np.int64)
        for i in range(n_groups)
    ]

    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for combo in combinations(range(n_groups), n_test_groups):
        oos_idx = np.concatenate([groups[i] for i in combo])
        is_mask = np.ones(n_obs, dtype=bool)
        for i in combo:
            lo = max(boundaries[i] - embargo, 0)
            hi = min(boundaries[i + 1] + embargo, n_obs)
            is_mask[lo:hi] = False
        is_idx = np.flatnonzero(is_mask).astype(np.int64)
        splits.append((is_idx, oos_idx))
    return splits


def score_grid(
    returns_by_strategy: dict[str, NDArray[np.float64]],
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    *,
    metric_fn,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Score every strategy on every split.

    Returns (is_perf, oos_perf, strategy_names) where ``is_perf`` and
    ``oos_perf`` are arrays of shape (n_splits, n_strategies) suitable for
    ``probability_of_backtest_overfitting``.
    """
    names = list(returns_by_strategy.keys())
    n_splits = len(splits)
    n_strats = len(names)
    is_perf = np.zeros((n_splits, n_strats), dtype=np.float64)
    oos_perf = np.zeros((n_splits, n_strats), dtype=np.float64)
    for s_idx, (is_idx, oos_idx) in enumerate(splits):
        for j, name in enumerate(names):
            r = returns_by_strategy[name]
            is_perf[s_idx, j] = metric_fn(r[is_idx])
            oos_perf[s_idx, j] = metric_fn(r[oos_idx])
    return is_perf, oos_perf, names
