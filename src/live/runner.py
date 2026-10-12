"""Live (or shadow) trading runner — pure-Python, deterministic given inputs.

The runner drives one trading session. It does NOT contain its own clock or
async loop — the deployment script (run_live, run_shadow) supplies inputs
and decides when to call. This makes the runner trivially testable.

Flow per call to ``run_session``:
  1. Read account + positions.
  2. Reset kill switch.
  3. Compute regime multiplier.
  4. Generate signals from each enabled component.
  5. Allocate sizes; submit via OrderManager.
  6. Update bot_state.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from src.backtest.engine import DayInputs
from src.ensemble.allocator import AllocatorParams, apply_allocation
from src.ensemble.kill_switch import KillSwitch, KillSwitchState
from src.live.alpaca_client import BrokerClient
from src.live.order_manager import OrderManager
from src.live.state import BotState, write_state
from src.strategies.base import Trade
from src.strategies.intraday_momentum import (
    IntradayMomentumParams,
    intraday_momentum_signals,
)
from src.strategies.orb_stocks_in_play import ORBParams, orb_signals
from src.strategies.vol_regime import VolRegimeParams, regime_multiplier
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RunnerConfig:
    state_path: Path = Path("bot_state.json")
    daily_loss_limit: float = 0.025
    weekly_loss_limit: float = 0.05
    orb_params: ORBParams = field(default_factory=ORBParams)
    momentum_params: IntradayMomentumParams = field(default_factory=IntradayMomentumParams)
    vol_regime_params: VolRegimeParams = field(default_factory=VolRegimeParams)
    allocator_params: AllocatorParams = field(default_factory=AllocatorParams)
    enable_orb: bool = True
    enable_intraday_momentum: bool = True


@dataclass
class Runner:
    broker: BrokerClient
    cfg: RunnerConfig = field(default_factory=RunnerConfig)
    state: BotState = field(default_factory=BotState)
    _kill_switch: KillSwitch = field(init=False)
    _om: OrderManager = field(init=False)

    def __post_init__(self) -> None:
        self._kill_switch = KillSwitch(
            daily_loss_limit=self.cfg.daily_loss_limit,
            weekly_loss_limit=self.cfg.weekly_loss_limit,
        )
        self._om = OrderManager(broker=self.broker)

    def run_session(self, day: DayInputs) -> list[Trade]:
        """Execute one full session (called once per trading day)."""
        acct = self.broker.account()
        equity = acct.equity
        self.state.equity = equity
        self.state.cash = acct.cash

        self._kill_switch.reset_for_day(day.session_date, equity)
        self.state.kill_switch_state = self._kill_switch.state.value
        if not self._kill_switch.can_trade():
            self.state.last_event = "halted"
            write_state(self.state, self.cfg.state_path)
            log.info("runner.halted", extra={"state": self.state.kill_switch_state})
            return []

        mult = regime_multiplier(
            day.vix_today,
            day.vix_history,
            day.spy_close_history,
            self.cfg.vol_regime_params,
        )
        self.state.regime_multiplier = mult
        if mult <= 0:
            self.state.last_event = "regime_halt"
            write_state(self.state, self.cfg.state_path)
            log.info("runner.regime_halt")
            return []

        intents: list[Trade] = []
        if self.cfg.enable_orb:
            intents += orb_signals(
                day.minute_bars_orb,
                day.daily_bars_orb,
                day.session_date,
                equity=equity,
                params=self.cfg.orb_params,
            )
        if self.cfg.enable_intraday_momentum:
            intents += intraday_momentum_signals(
                day.minute_bars_mom,
                day.prev_close_mom,
                day.session_date,
                equity=equity,
                params=self.cfg.momentum_params,
            )

        sized = apply_allocation(
            intents,
            regime_multiplier=mult,
            params=self.cfg.allocator_params,
        )

        for t in sized:
            self._om.submit(t, day.session_date)

        self.state.today_trades = len(sized)
        self.state.last_event = "submitted"
        positions = self.broker.positions()
        self.state.open_positions = [
            {"symbol": p.symbol, "qty": p.qty, "avg_entry_price": p.avg_entry_price}
            for p in positions
        ]
        write_state(self.state, self.cfg.state_path)
        log.info("runner.session_done", extra={"n_trades": len(sized)})
        return sized

    def end_of_day_flatten(self) -> None:
        self._om.flatten_all()
        self.state.last_event = "eod_flat"
        write_state(self.state, self.cfg.state_path)

    def manual_reenable(self) -> None:
        self._kill_switch.manual_reenable()
        self.state.kill_switch_state = self._kill_switch.state.value
        write_state(self.state, self.cfg.state_path)
