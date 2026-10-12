from __future__ import annotations

from datetime import date, datetime, time, timezone

from src.utils.clock import (
    EOD_FLAT,
    EOD_FLAT_HALF,
    NY_TZ,
    force_flat_time,
    in_rth,
    is_half_day,
    is_holiday,
    is_trading_day,
    opening_range_window,
    session_info,
    to_ny,
    to_utc,
    trading_days,
)


def test_to_ny_handles_naive_as_utc() -> None:
    naive = datetime(2025, 6, 2, 13, 30)  # 13:30 UTC == 09:30 EDT
    ny = to_ny(naive)
    assert ny.tzinfo == NY_TZ
    assert ny.hour == 9 and ny.minute == 30


def test_to_utc_assumes_ny_for_naive() -> None:
    naive = datetime(2025, 6, 2, 9, 30)
    utc = to_utc(naive)
    assert utc.tzinfo == timezone.utc
    assert utc.hour == 13 and utc.minute == 30


def test_is_trading_day_rejects_weekends() -> None:
    assert not is_trading_day(date(2025, 6, 7))  # sat
    assert not is_trading_day(date(2025, 6, 8))  # sun
    assert is_trading_day(date(2025, 6, 9))


def test_is_holiday_july_4_2025() -> None:
    assert is_holiday(date(2025, 7, 4))
    assert not is_trading_day(date(2025, 7, 4))


def test_half_day_uses_1255_flat_time() -> None:
    d = date(2024, 11, 29)  # Black Friday 2024
    assert is_half_day(d)
    assert force_flat_time(d) == EOD_FLAT_HALF
    assert force_flat_time(date(2025, 6, 9)) == EOD_FLAT


def test_in_rth_boundary() -> None:
    open_dt = datetime(2025, 6, 9, 9, 30, tzinfo=NY_TZ)
    just_before = datetime(2025, 6, 9, 9, 29, 59, tzinfo=NY_TZ)
    just_after_close = datetime(2025, 6, 9, 16, 0, tzinfo=NY_TZ)
    assert in_rth(open_dt)
    assert not in_rth(just_before)
    assert not in_rth(just_after_close)


def test_session_info_full_day() -> None:
    info = session_info(date(2025, 6, 9))
    assert info is not None
    assert info.open_dt_ny.time() == time(9, 30)
    assert info.close_dt_ny.time() == time(16, 0)
    assert not info.is_half_day
    assert not info.is_holiday


def test_session_info_holiday_marked() -> None:
    info = session_info(date(2025, 7, 4))
    assert info is not None and info.is_holiday


def test_trading_days_excludes_weekend_and_holiday() -> None:
    days = trading_days(date(2025, 7, 1), date(2025, 7, 7))
    # 7/4 is Friday (Independence Day), 7/5-6 weekend
    assert date(2025, 7, 4) not in days
    assert date(2025, 7, 5) not in days
    assert date(2025, 7, 1) in days


def test_opening_range_window_is_five_minutes() -> None:
    s, e = opening_range_window(date(2025, 6, 9))
    assert (e - s).total_seconds() == 300
    assert s.time() == time(9, 30)
    assert e.time() == time(9, 35)
