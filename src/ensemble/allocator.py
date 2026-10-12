"""Capital allocator — splits the risk budget across components.

Default 60/30/10:
    60% of risk budget for ORB Stocks-in-Play
    30% of risk budget for Intraday Momentum
    10% always-cash reserve

The vol-regime multiplier is applied multiplicatively *after* the static
weights — so in a halved-size regime the components still trade at their
relative ratios.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.strategies.base import Trade


@dataclass(frozen=True)
class AllocatorParams:
    weight_orb: float = 0.60
    weight_intraday_momentum: float = 0.30
    cash_reserve: float = 0.10

    def __post_init__(self) -> None:
        total = self.weight_orb + self.weight_intraday_momentum + self.cash_reserve
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"allocator weights must sum to 1.0, got {total:.6f}"
            )


def apply_allocation(
    trades: list[Trade],
    *,
    regime_multiplier: float,
    params: AllocatorParams = AllocatorParams(),
) -> list[Trade]:
    """Return trades resized by component-weight * regime multiplier.

    Trades whose post-scaling share count rounds to 0 are dropped.
    """
    if regime_multiplier <= 0:
        return []

    out: list[Trade] = []
    for t in trades:
        if t.component == "orb":
            w = params.weight_orb
        elif t.component == "intraday_momentum":
            w = params.weight_intraday_momentum
        else:
            continue
        scaled = int(t.shares * w * regime_multiplier)
        if scaled <= 0:
            continue
        out.append(
            Trade(
                symbol=t.symbol,
                side=t.side,
                entry_time=t.entry_time,
                entry_price=t.entry_price,
                stop_price=t.stop_price,
                shares=scaled,
                exit_time=t.exit_time,
                component=t.component,
                take_profit_price=t.take_profit_price,
                meta=t.meta,
            )
        )
    return out
