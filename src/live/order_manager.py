"""Idempotent order submission.

Each Trade gets a deterministic client_order_id derived from
(component, symbol, session_date) so that retries in the face of network
hiccups never produce duplicate fills.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

from src.live.alpaca_client import BrokerClient
from src.strategies.base import Trade
from src.utils.logging import get_logger

log = get_logger(__name__)


def make_client_order_id(trade: Trade, session_date: date) -> str:
    return f"nt-{trade.component}-{trade.symbol}-{session_date.isoformat()}"


@dataclass
class OrderManager:
    broker: BrokerClient
    submitted: dict[str, str] = field(default_factory=dict)
    """Map of client_order_id -> broker order id."""

    def submit(self, trade: Trade, session_date: date) -> str:
        coid = make_client_order_id(trade, session_date)
        if coid in self.submitted:
            log.info("om.duplicate_skipped", extra={"coid": coid})
            return self.submitted[coid]
        if trade.component == "orb":
            broker_id = self.broker.submit_bracket_long(
                symbol=trade.symbol,
                shares=int(trade.shares),
                limit_price=float(trade.entry_price),
                stop_price=float(trade.stop_price),
                client_order_id=coid,
            )
        else:
            broker_id = self.broker.submit_market_long(
                symbol=trade.symbol,
                shares=int(trade.shares),
                client_order_id=coid,
            )
        self.submitted[coid] = broker_id
        log.info(
            "om.submitted",
            extra={
                "coid": coid,
                "broker_id": broker_id,
                "symbol": trade.symbol,
                "shares": trade.shares,
                "component": trade.component,
            },
        )
        return broker_id

    def flatten_all(self) -> None:
        log.info("om.flatten_all")
        self.broker.flatten_all()
