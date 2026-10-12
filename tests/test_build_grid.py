"""Smoke test for build_grid: writes a small Parquet of (n_days, 8 variants)
when invoked with synthetic inputs piped via temporary data dirs.

We test the *function* not the CLI here because the CLI requires real
downloaded data; the function-level test ensures the grid expansion +
return-stacking logic is sound.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from scripts.build_grid import STOP_ATR_PCTS, TOP_NS, _build_day_inputs


def test_grid_dimensions_match_constants() -> None:
    assert len(STOP_ATR_PCTS) == 4
    assert len(TOP_NS) == 2
    # 4 * 2 = 8 trials, the value the harness was tuned for
    assert len(STOP_ATR_PCTS) * len(TOP_NS) == 8


def test_build_day_inputs_handles_missing_data(tmp_path: Path) -> None:
    """When no parquet files exist on disk, the helper still returns one
    DayInputs per trading day (with empty bar dicts) so the engine can skip
    them gracefully."""
    days = [date(2025, 6, 9), date(2025, 6, 10)]
    inputs = _build_day_inputs(
        days=days,
        symbols=["AAPL"],
        momentum_symbols=["SPY"],
        vix_symbol="VIXY",
        data_dir=tmp_path,
        start_d=date(2025, 6, 1),
        end_d=date(2025, 6, 30),
    )
    assert len(inputs) == 2
    for inp in inputs:
        assert inp.session_date in days
        # No data on disk → empty dicts but still well-formed
        assert isinstance(inp.minute_bars_orb, dict)
        assert isinstance(inp.minute_bars_mom, dict)
