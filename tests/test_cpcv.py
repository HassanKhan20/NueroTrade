from __future__ import annotations

import numpy as np

from src.backtest.cpcv import cpcv_splits, score_grid


def test_split_count_matches_combinations() -> None:
    # C(6, 2) = 15
    splits = cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=0)
    assert len(splits) == 15


def test_no_overlap_between_is_and_oos() -> None:
    splits = cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=0)
    for is_idx, oos_idx in splits:
        assert set(is_idx).isdisjoint(set(oos_idx))


def test_embargo_purges_neighbours() -> None:
    splits = cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=3)
    for is_idx, oos_idx in splits:
        # IS samples must be at least `embargo` away from any OOS sample
        oos_set = set(oos_idx.tolist())
        for i in is_idx:
            for o in (i - 1, i - 2, i - 3, i + 1, i + 2, i + 3):
                assert o not in oos_set


def test_score_grid_shapes() -> None:
    n = 240
    rng = np.random.default_rng(0)
    rets = {f"s{i}": rng.normal(0, 0.01, n) for i in range(4)}
    splits = cpcv_splits(n_obs=n, n_groups=6, n_test_groups=2, embargo=0)
    is_perf, oos_perf, names = score_grid(rets, splits, metric_fn=np.mean)
    assert is_perf.shape == (15, 4)
    assert oos_perf.shape == (15, 4)
    assert names == ["s0", "s1", "s2", "s3"]
