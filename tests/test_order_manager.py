from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time

import pytest

from src.live.alpaca_client import AccountSnapshot, Position
from src.live.order_manager import OrderManager, make_client_order_id
from src.strategies.base import Side, Trade


@dataclass
class FakeBroker:
    submitted: list[tuple[str, str, int]] = field(default_factory=list)

    def account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=100_000.0, cash=100_000.0, buying_power=100_000.0)

    def positions(self) -> list[Position]:
        return []

    def submit_bracket_long(self, symbol, shares, limit_price, stop_price, client_order_id):
        self.submitted.append(("bracket", client_order_id, shares))
        return f"broker-{len(self.submitted)}"

    def submit_market_long(self, symbol, shares, client_order_id):
        self.submitted.append(("market", client_order_id, shares))
        return f"broker-{len(self.submitted)}"

    def close_position(self, symbol):
        self.submitted.append(("close", symbol, 0))

    def flatten_all(self):
        self.submitted.append(("flatten", "", 0))


def _trade(component: str) -> Trade:
    return Trade(
        symbol="AAPL",
        side=Side.LONG,
        entry_time=datetime(2025, 6, 9, 9, 35),
        entry_price=200.0,
        stop_price=199.5,
        shares=100,
        exit_time=datetime(2025, 6, 9, 15, 55),
        component=component,
    )


def test_make_client_order_id_deterministic() -> None:
    t = _trade("orb")
    assert make_client_order_id(t, date(2025, 6, 9)) == "nt-orb-AAPL-2025-06-09"


def test_orb_uses_bracket_path() -> None:
    b = FakeBroker()
    om = OrderManager(broker=b)
    om.submit(_trade("orb"), date(2025, 6, 9))
    assert b.submitted[0][0] == "bracket"


def test_momentum_uses_market_path() -> None:
    b = FakeBroker()
    om = OrderManager(broker=b)
    om.submit(_trade("intraday_momentum"), date(2025, 6, 9))
    assert b.submitted[0][0] == "market"


def test_idempotent_submit_same_day() -> None:
    b = FakeBroker()
    om = OrderManager(broker=b)
    om.submit(_trade("orb"), date(2025, 6, 9))
    om.submit(_trade("orb"), date(2025, 6, 9))
    assert len(b.submitted) == 1


def test_flatten_all_calls_broker() -> None:
    b = FakeBroker()
    om = OrderManager(broker=b)
    om.flatten_all()
    assert b.submitted == [("flatten", "", 0)]
