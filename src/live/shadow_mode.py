"""Shadow mode broker — looks like a BrokerClient, never sends real orders.

Every "submission" is appended to logs/shadow_orders.jsonl so the next
backtest run can compare its simulated fills against the shadow log.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.live.alpaca_client import AccountSnapshot, Position
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ShadowBroker:
    log_path: Path
    starting_equity: float = 25_000.0
    _equity: float = field(init=False)
    _positions: list[Position] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._equity = self.starting_equity
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=self._equity, cash=self._equity, buying_power=self._equity
        )

    def positions(self) -> list[Position]:
        return list(self._positions)

    def _record(self, payload: dict) -> str:
        payload = {
            **payload,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
        return payload.get("client_order_id", str(uuid4()))

    def submit_bracket_long(
        self, symbol, shares, limit_price, stop_price, client_order_id
    ):
        bid = f"shadow-{uuid4().hex[:8]}"
        self._record(
            {
                "type": "bracket_long",
                "symbol": symbol,
                "shares": shares,
                "limit_price": limit_price,
                "stop_price": stop_price,
                "client_order_id": client_order_id,
                "broker_id": bid,
            }
        )
        return bid

    def submit_market_long(self, symbol, shares, client_order_id):
        bid = f"shadow-{uuid4().hex[:8]}"
        self._record(
            {
                "type": "market_long",
                "symbol": symbol,
                "shares": shares,
                "client_order_id": client_order_id,
                "broker_id": bid,
            }
        )
        return bid

    def close_position(self, symbol: str) -> None:
        self._record({"type": "close_position", "symbol": symbol})
        self._positions = [p for p in self._positions if p.symbol != symbol]

    def flatten_all(self) -> None:
        self._record({"type": "flatten_all"})
        self._positions = []
