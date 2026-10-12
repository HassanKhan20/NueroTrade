"""Common types shared by all strategy primitives.

This bot is **long-only**. The Side enum still exists so legacy code paths
have explicit names, but Side.SHORT must never appear in an emitted Trade.
The Trade dataclass enforces this in __post_init__.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(Enum):
    LONG = 1
    SHORT = -1  # retained for clarity in signal logic; never produced as output.

    @property
    def sign(self) -> int:
        return self.value


@dataclass(frozen=True)
class Trade:
    """A single round-trip trade as emitted by a strategy primitive.

    Strategies emit Trades as *intent*; the backtest engine and live runner
    decide how to send them. Exit precedence: take_profit (if set) > stop > time.

    Long-only invariant: ``side`` must be ``Side.LONG``, the stop must be
    strictly below the entry, and any take-profit must be strictly above
    the entry. Constructing a SHORT Trade raises ValueError. Equities only.
    """

    symbol: str
    side: Side
    entry_time: datetime
    entry_price: float
    stop_price: float
    shares: int
    exit_time: datetime  # forced flat by this NY-time wall clock
    component: str  # "orb" | "intraday_momentum"
    take_profit_price: float | None = None
    meta: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.side is not Side.LONG:
            raise ValueError(
                f"Long-only bot: Side.SHORT trades are not permitted "
                f"(symbol={self.symbol}, component={self.component})"
            )
        if self.shares <= 0:
            raise ValueError(f"shares must be > 0, got {self.shares}")
        if self.stop_price >= self.entry_price:
            raise ValueError(
                f"Long stop must be strictly below entry: "
                f"entry={self.entry_price}, stop={self.stop_price}"
            )
        if self.take_profit_price is not None and self.take_profit_price <= self.entry_price:
            raise ValueError(
                f"Long take-profit must be strictly above entry: "
                f"entry={self.entry_price}, tp={self.take_profit_price}"
            )

    def stop_distance(self) -> float:
        return self.entry_price - self.stop_price  # always > 0 by invariant

    def risk_dollars(self) -> float:
        return self.stop_distance() * self.shares
