"""Thin Alpaca client wrapper.

Behind a Protocol so the order manager can be tested without alpaca-py
present. The real implementation is constructed lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float


@dataclass
class Position:
    symbol: str
    qty: int
    avg_entry_price: float


class BrokerClient(Protocol):
    def account(self) -> AccountSnapshot: ...
    def positions(self) -> list[Position]: ...
    def submit_bracket_long(
        self,
        symbol: str,
        shares: int,
        limit_price: float | None,
        stop_price: float,
        client_order_id: str,
    ) -> str: ...
    def submit_market_long(
        self, symbol: str, shares: int, client_order_id: str
    ) -> str: ...
    def close_position(self, symbol: str) -> None: ...
    def flatten_all(self) -> None: ...


class AlpacaBrokerClient:
    """Production client. Imports alpaca-py lazily."""

    def __init__(self, key: str, secret: str, paper: bool = True) -> None:
        if not key or not secret:
            raise ValueError("Alpaca credentials missing")
        self._key = key
        self._secret = secret
        self._paper = paper
        self._client = None

    def _ensure(self) -> None:
        if self._client is not None:
            return
        from alpaca.trading.client import TradingClient  # type: ignore

        self._client = TradingClient(self._key, self._secret, paper=self._paper)

    def account(self) -> AccountSnapshot:
        self._ensure()
        a = self._client.get_account()  # type: ignore[union-attr]
        return AccountSnapshot(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
        )

    def positions(self) -> list[Position]:
        self._ensure()
        out: list[Position] = []
        for p in self._client.get_all_positions():  # type: ignore[union-attr]
            out.append(
                Position(
                    symbol=str(p.symbol),
                    qty=int(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                )
            )
        return out

    def submit_bracket_long(
        self,
        symbol: str,
        shares: int,
        limit_price: float | None,
        stop_price: float,
        client_order_id: str,
    ) -> str:
        self._ensure()
        from alpaca.trading.requests import (  # type: ignore
            LimitOrderRequest,
            MarketOrderRequest,
            StopLossRequest,
        )
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce  # type: ignore

        if limit_price is not None:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.OTO,
                limit_price=limit_price,
                stop_loss=StopLossRequest(stop_price=stop_price),
                client_order_id=client_order_id,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.OTO,
                stop_loss=StopLossRequest(stop_price=stop_price),
                client_order_id=client_order_id,
            )
        order = self._client.submit_order(req)  # type: ignore[union-attr]
        return str(order.id)

    def submit_market_long(
        self, symbol: str, shares: int, client_order_id: str
    ) -> str:
        self._ensure()
        from alpaca.trading.requests import MarketOrderRequest  # type: ignore
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore

        req = MarketOrderRequest(
            symbol=symbol,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = self._client.submit_order(req)  # type: ignore[union-attr]
        return str(order.id)

    def close_position(self, symbol: str) -> None:
        self._ensure()
        self._client.close_position(symbol)  # type: ignore[union-attr]

    def flatten_all(self) -> None:
        self._ensure()
        self._client.close_all_positions(cancel_orders=True)  # type: ignore[union-attr]
