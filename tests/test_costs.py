from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from src.costs.commissions import commission, round_trip_cost
from src.costs.slippage import fill_price, slippage_amount


# --- commissions -----------------------------------------------------------
def test_commission_zero_shares_is_zero() -> None:
    assert commission(0) == 0.0


def test_commission_min_ticket_floor() -> None:
    # 50 shares * $0.0035 = $0.175 -> floor at $0.35
    assert commission(50) == pytest.approx(0.35)


def test_commission_above_floor_uses_per_share() -> None:
    # 1000 shares * $0.0035 = $3.50, well above floor
    assert commission(1000) == pytest.approx(3.50)


def test_commission_sign_agnostic() -> None:
    assert commission(-200) == commission(200)


@given(shares=st.integers(min_value=1, max_value=100_000))
def test_commission_monotonic_in_shares(shares: int) -> None:
    assert commission(shares + 1) >= commission(shares)


def test_round_trip_is_double() -> None:
    assert round_trip_cost(500) == pytest.approx(2 * commission(500))


# --- slippage --------------------------------------------------------------
def test_buy_fills_above_ref() -> None:
    fp = fill_price(+1, 100.0, {"high": 101.0, "low": 99.0})
    assert fp > 100.0


def test_sell_fills_below_ref() -> None:
    fp = fill_price(-1, 100.0, {"high": 101.0, "low": 99.0})
    assert fp < 100.0


def test_slippage_amount_nonnegative() -> None:
    assert slippage_amount(+1, 50.0, {"high": 50.5, "low": 49.5}) >= 0
    assert slippage_amount(-1, 50.0, {"high": 50.5, "low": 49.5}) >= 0


def test_slippage_grows_with_bar_range() -> None:
    narrow = slippage_amount(+1, 50.0, {"high": 50.05, "low": 49.95})
    wide = slippage_amount(+1, 50.0, {"high": 51.0, "low": 49.0})
    assert wide > narrow


def test_invalid_side_raises() -> None:
    with pytest.raises(ValueError):
        fill_price(0, 100.0, {"high": 101.0, "low": 99.0})
