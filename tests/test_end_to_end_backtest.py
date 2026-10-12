"""End-to-end backtest with synthetic data: confirms the full pipeline from
strategy primitives through the engine through reports works on multi-day
data without any real bars on disk.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestConfig, Backtester, DayInputs
from src.backtest.reports import summarise, write_report
from src.utils.clock import trading_days
from tests.fixtures.synthetic import (
    daily_from_minute,
    synth_minute_history,
)


def _build_inputs_for_period(
    start: date, end: date, *, drift: float, range_pct: float
) -> list[DayInputs]:
    days = trading_days(start, end)
    aapl_minute = synth_minute_history(
        "AAPL", days, open_price=180.0,
        drift_per_min=drift, range_pct=range_pct,
    )
    spy_minute = synth_minute_history(
        "SPY", days, open_price=500.0,
        drift_per_min=drift * 0.5, range_pct=range_pct,
    )
    aapl_daily = daily_from_minute(aapl_minute)
    spy_daily = daily_from_minute(spy_minute)
    inputs: list[DayInputs] = []
    for d in days:
        day_minute_aapl = aapl_minute[
            (aapl_minute.index.date == d)
        ]
        day_minute_spy = spy_minute[
            (spy_minute.index.date == d)
        ]
        prior_spy = spy_daily.loc[spy_daily.index.date < d]
        prev_close_spy = (
            float(prior_spy["close"].iloc[-1]) if not prior_spy.empty else 500.0
        )
        inputs.append(
            DayInputs(
                session_date=d,
                minute_bars_orb={"AAPL": day_minute_aapl},
                daily_bars_orb={"AAPL": aapl_daily.loc[aapl_daily.index.date < d]},
                minute_bars_mom={"SPY": day_minute_spy},
                prev_close_mom={"SPY": prev_close_spy},
                vix_today=18.0,
                vix_history=pd.Series(dtype=float),
                spy_close_history=pd.Series(dtype=float),
            )
        )
    return inputs


def test_engine_produces_equity_curve_and_trades() -> None:
    inputs = _build_inputs_for_period(
        date(2025, 1, 6), date(2025, 1, 31),  # ~20 trading days
        drift=0.005, range_pct=0.0008,
    )
    bt = Backtester(BacktestConfig(initial_equity=100_000.0))
    result = bt.run(inputs)

    assert len(result.equity_curve) == len(inputs)
    # In a uniformly-up day, ORB should fire on most days.
    assert result.n_trades() > 0
    # Daily returns must have the same length as equity curve.
    assert len(result.daily_returns) == len(result.equity_curve)
    # Component attribution dict has both keys.
    assert set(result.component_attribution.keys()) >= {"orb", "intraday_momentum"}


def test_summarise_returns_finite_metrics() -> None:
    inputs = _build_inputs_for_period(
        date(2025, 2, 3), date(2025, 2, 28),
        drift=0.003, range_pct=0.0006,
    )
    bt = Backtester(BacktestConfig(initial_equity=100_000.0))
    s = summarise(bt.run(inputs))
    assert s["n_days"] > 0
    assert isinstance(s["sharpe"], float)
    assert s["terminal_equity"] > 0


def test_report_writes_csvs_and_json(tmp_path: Path) -> None:
    inputs = _build_inputs_for_period(
        date(2025, 3, 3), date(2025, 3, 14),
        drift=0.002, range_pct=0.0005,
    )
    bt = Backtester(BacktestConfig(initial_equity=100_000.0))
    out = write_report(bt.run(inputs), tmp_path, label="e2e")
    assert out.exists()
    assert (tmp_path / "equity_curve.csv").exists()
    assert (tmp_path / "trades.csv").exists()
