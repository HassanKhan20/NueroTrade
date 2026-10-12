from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.engine import DayInputs
from src.live.alpaca_client import AccountSnapshot, Position
from src.live.runner import Runner, RunnerConfig
from src.live.state import read_state


@dataclass
class StubBroker:
    equity: float = 100_000.0
    submitted: list[tuple] = field(default_factory=list)

    def account(self):
        return AccountSnapshot(equity=self.equity, cash=self.equity, buying_power=self.equity)

    def positions(self):
        return []

    def submit_bracket_long(self, symbol, shares, limit_price, stop_price, client_order_id):
        self.submitted.append(("bracket", symbol, shares))
        return f"id-{len(self.submitted)}"

    def submit_market_long(self, symbol, shares, client_order_id):
        self.submitted.append(("market", symbol, shares))
        return f"id-{len(self.submitted)}"

    def close_position(self, symbol):
        return None

    def flatten_all(self):
        self.submitted.append(("flatten",))


def _empty_day(d: date) -> DayInputs:
    return DayInputs(
        session_date=d,
        minute_bars_orb={},
        daily_bars_orb={},
        minute_bars_mom={},
        prev_close_mom={},
        vix_today=18.0,
        vix_history=pd.Series(dtype=float),
        spy_close_history=pd.Series(dtype=float),
    )


def test_runner_writes_state(tmp_path: Path) -> None:
    state_path = tmp_path / "bot_state.json"
    runner = Runner(broker=StubBroker(), cfg=RunnerConfig(state_path=state_path))
    runner.run_session(_empty_day(date(2025, 6, 9)))
    s = read_state(state_path)
    assert s is not None
    assert s.equity == 100_000.0
    assert s.kill_switch_state == "ok"


def test_runner_halts_on_holiday_no_signals(tmp_path: Path) -> None:
    state_path = tmp_path / "bot_state.json"
    runner = Runner(broker=StubBroker(), cfg=RunnerConfig(state_path=state_path))
    out = runner.run_session(_empty_day(date(2025, 7, 4)))
    assert out == []


def test_runner_kill_switch_blocks_trading(tmp_path: Path) -> None:
    state_path = tmp_path / "bot_state.json"
    broker = StubBroker(equity=100_000.0)
    runner = Runner(broker=broker, cfg=RunnerConfig(state_path=state_path))
    runner.run_session(_empty_day(date(2025, 6, 9)))

    # Drop equity below daily limit and run again
    broker.equity = 95_000.0
    runner._kill_switch.update_equity(95_000.0)
    out = runner.run_session(_empty_day(date(2025, 6, 9)))  # same day -> halted
    assert out == []
    s = read_state(state_path)
    assert s is not None
    # Either daily_halted (same day) or weekly_halted (if cumulative weekly breach)
    assert s.kill_switch_state in {"daily_halted", "weekly_halted"}


def test_runner_flatten(tmp_path: Path) -> None:
    state_path = tmp_path / "bot_state.json"
    broker = StubBroker()
    runner = Runner(broker=broker, cfg=RunnerConfig(state_path=state_path))
    runner.end_of_day_flatten()
    assert ("flatten",) in broker.submitted
