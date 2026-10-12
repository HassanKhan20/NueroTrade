from __future__ import annotations

from datetime import datetime

import pytest

from src.ensemble.allocator import AllocatorParams, apply_allocation
from src.strategies.base import Side, Trade


def _trade(symbol: str, component: str, shares: int) -> Trade:
    return Trade(
        symbol=symbol,
        side=Side.LONG,
        entry_time=datetime(2025, 6, 9, 9, 35),
        entry_price=100.0,
        stop_price=99.5,
        shares=shares,
        exit_time=datetime(2025, 6, 9, 15, 55),
        component=component,
    )


def test_invalid_weights_raise() -> None:
    with pytest.raises(ValueError):
        AllocatorParams(weight_orb=0.5, weight_intraday_momentum=0.5, cash_reserve=0.5)


def test_default_60_30_10_split() -> None:
    trades = [_trade("AAPL", "orb", 1000), _trade("QQQ", "intraday_momentum", 1000)]
    out = apply_allocation(trades, regime_multiplier=1.0)
    by_comp = {t.component: t for t in out}
    assert by_comp["orb"].shares == 600
    assert by_comp["intraday_momentum"].shares == 300


def test_regime_halve_applied() -> None:
    trades = [_trade("AAPL", "orb", 1000)]
    out = apply_allocation(trades, regime_multiplier=0.5)
    assert out[0].shares == 300


def test_regime_zero_drops_all() -> None:
    trades = [_trade("AAPL", "orb", 1000)]
    out = apply_allocation(trades, regime_multiplier=0.0)
    assert out == []


def test_drops_zero_size_after_scaling() -> None:
    trades = [_trade("AAPL", "intraday_momentum", 2)]
    out = apply_allocation(trades, regime_multiplier=0.1)
    # 2 * 0.30 * 0.1 = 0.06 -> int -> 0 -> dropped
    assert out == []


def test_unknown_component_dropped() -> None:
    trades = [_trade("AAPL", "mystery", 100)]
    out = apply_allocation(trades, regime_multiplier=1.0)
    assert out == []


def test_take_profit_propagates_through_allocator() -> None:
    t = Trade(
        symbol="X",
        side=Side.LONG,
        entry_time=datetime(2025, 6, 9, 9, 35),
        entry_price=100.0,
        stop_price=99.5,
        take_profit_price=101.5,
        shares=1000,
        exit_time=datetime(2025, 6, 9, 15, 55),
        component="orb",
    )
    out = apply_allocation([t], regime_multiplier=1.0)
    assert len(out) == 1
    assert out[0].take_profit_price == 101.5
