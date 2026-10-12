from __future__ import annotations

from datetime import date

import pytest

from src.ensemble.kill_switch import KillSwitch, KillSwitchState


def test_starts_ok() -> None:
    ks = KillSwitch()
    ks.reset_for_day(date(2025, 6, 9), 100_000.0)
    assert ks.state == KillSwitchState.OK
    assert ks.can_trade()


def test_daily_breach_halts_for_day() -> None:
    ks = KillSwitch(daily_loss_limit=0.025)
    ks.reset_for_day(date(2025, 6, 9), 100_000.0)
    state = ks.update_equity(97_500.0 - 1.0)  # just past -2.5%
    assert state == KillSwitchState.DAILY_HALTED
    assert not ks.can_trade()


def test_daily_halt_clears_next_day() -> None:
    ks = KillSwitch(daily_loss_limit=0.025)
    ks.reset_for_day(date(2025, 6, 9), 100_000.0)
    ks.update_equity(97_000.0)
    assert ks.state == KillSwitchState.DAILY_HALTED
    ks.reset_for_day(date(2025, 6, 10), 97_000.0)
    assert ks.state == KillSwitchState.OK
    assert ks.can_trade()


def test_weekly_breach_persists_until_manual_reenable() -> None:
    ks = KillSwitch(daily_loss_limit=0.99, weekly_loss_limit=0.05)
    ks.reset_for_day(date(2025, 6, 9), 100_000.0)  # Mon
    state = ks.update_equity(94_000.0)             # -6%
    assert state == KillSwitchState.WEEKLY_HALTED
    # next day still halted
    ks.reset_for_day(date(2025, 6, 10), 94_000.0)
    assert ks.state == KillSwitchState.WEEKLY_HALTED
    assert not ks.can_trade()
    # manual re-enable
    ks.manual_reenable()
    assert ks.can_trade()


def test_no_breach_at_threshold_floor() -> None:
    ks = KillSwitch(daily_loss_limit=0.025)
    ks.reset_for_day(date(2025, 6, 9), 100_000.0)
    # exactly -2.4999%
    state = ks.update_equity(97_500.01)
    assert state == KillSwitchState.OK
