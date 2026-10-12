"""Daily and weekly loss limits — fail-closed risk discipline.

The kill switch is checked before every signal generation step. If the
breach is intraday, the bot flattens immediately and rejects new orders for
the rest of the session. If weekly, it halts until ``enabled`` is flipped
back to True manually (e.g. by editing bot_state.json).

This is the single most important piece of code in the whole bot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class KillSwitchState(Enum):
    OK = "ok"
    DAILY_HALTED = "daily_halted"
    WEEKLY_HALTED = "weekly_halted"


@dataclass
class KillSwitch:
    """Tracks running PnL vs. configured limits.

    Equity should be passed in at the start-of-day and updated as it changes.
    The kill switch never modifies positions itself — it only reports state;
    the runner is responsible for flattening.
    """

    daily_loss_limit: float = 0.025  # fraction of start-of-day equity
    weekly_loss_limit: float = 0.05  # fraction of start-of-week equity
    enabled: bool = True

    _start_of_day_equity: float = 0.0
    _start_of_week_equity: float = 0.0
    _current_equity: float = 0.0
    _current_day: date | None = None
    _week_anchor: date | None = None
    _state: KillSwitchState = KillSwitchState.OK
    _halted_until_day: date | None = None

    def reset_for_day(self, today: date, equity: float) -> None:
        if self._week_anchor is None or _weeks_apart(self._week_anchor, today) >= 1:
            self._week_anchor = _start_of_week(today)
            self._start_of_week_equity = equity
        self._current_day = today
        self._start_of_day_equity = equity
        self._current_equity = equity
        if self._state == KillSwitchState.DAILY_HALTED:
            # daily halt clears overnight unless weekly is also breached
            if self._weekly_breached(equity):
                self._state = KillSwitchState.WEEKLY_HALTED
            else:
                self._state = KillSwitchState.OK

    def update_equity(self, equity: float) -> KillSwitchState:
        self._current_equity = equity
        if not self.enabled:
            self._state = KillSwitchState.WEEKLY_HALTED
            return self._state
        if self._weekly_breached(equity):
            self._state = KillSwitchState.WEEKLY_HALTED
            self.enabled = False  # require manual re-enable
        elif self._daily_breached(equity):
            if self._state != KillSwitchState.WEEKLY_HALTED:
                self._state = KillSwitchState.DAILY_HALTED
        return self._state

    def can_trade(self) -> bool:
        return self.enabled and self._state == KillSwitchState.OK

    @property
    def state(self) -> KillSwitchState:
        return self._state

    def manual_reenable(self) -> None:
        """Operator-only: flip the kill switch back on after a weekly halt."""
        self.enabled = True
        self._state = KillSwitchState.OK
        if self._current_day is not None:
            self._start_of_week_equity = self._current_equity
            self._start_of_day_equity = self._current_equity
            self._week_anchor = _start_of_week(self._current_day)

    def _daily_breached(self, equity: float) -> bool:
        if self._start_of_day_equity <= 0:
            return False
        loss_frac = (self._start_of_day_equity - equity) / self._start_of_day_equity
        return loss_frac >= self.daily_loss_limit

    def _weekly_breached(self, equity: float) -> bool:
        if self._start_of_week_equity <= 0:
            return False
        loss_frac = (self._start_of_week_equity - equity) / self._start_of_week_equity
        return loss_frac >= self.weekly_loss_limit


def _start_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _weeks_apart(a: date, b: date) -> int:
    return (_start_of_week(b) - _start_of_week(a)).days // 7
