"""Trading-clock utilities — all market logic uses America/New_York time.

Internal storage is UTC; conversion happens here. Half-days (1pm close) are
flagged so the strategies can force-flat at 12:55 instead of 15:55.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
UTC = timezone.utc

RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
HALF_DAY_CLOSE = time(13, 0)

# 5-minute opening range candle: 09:30-09:34:59
OR_START = time(9, 30)
OR_END = time(9, 35)

# Intraday-momentum (Gao et al.) timestamps
SIGNAL_TIME = time(14, 55)
ENTRY_TIME = time(15, 0)

# Force-flat times
EOD_FLAT = time(15, 55)
EOD_FLAT_HALF = time(12, 55)

# Hard-coded NYSE half-day calendar (early closes at 1pm).
# Source: NYSE annual schedules. Update yearly.
_HALF_DAYS: frozenset[date] = frozenset(
    {
        date(2016, 11, 25), date(2016, 12, 23),
        date(2017, 7, 3), date(2017, 11, 24),
        date(2018, 7, 3), date(2018, 11, 23), date(2018, 12, 24),
        date(2019, 7, 3), date(2019, 11, 29), date(2019, 12, 24),
        date(2020, 11, 27), date(2020, 12, 24),
        date(2021, 11, 26),
        date(2022, 11, 25),
        date(2023, 7, 3), date(2023, 11, 24),
        date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24),
        date(2025, 11, 28), date(2025, 12, 24),
        date(2026, 11, 27), date(2026, 12, 24),
    }
)

# Hard-coded NYSE full-closure holidays. Update yearly.
_FULL_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2024
        date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
        date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
        date(2024, 11, 28), date(2024, 12, 25),
        # 2025
        date(2025, 1, 1), date(2025, 1, 9), date(2025, 1, 20), date(2025, 2, 17),
        date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),
        date(2025, 9, 1), date(2025, 11, 27), date(2025, 12, 25),
        # 2026
        date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
        date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
        date(2026, 11, 26), date(2026, 12, 25),
    }
)


@dataclass(frozen=True)
class SessionInfo:
    session_date: date
    open_dt_ny: datetime
    close_dt_ny: datetime
    is_half_day: bool
    is_holiday: bool


def to_ny(dt: datetime) -> datetime:
    """Return dt in America/New_York. Naive input is treated as UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(NY_TZ)


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY_TZ)
    return dt.astimezone(UTC)


def is_holiday(d: date) -> bool:
    return d in _FULL_HOLIDAYS


def is_half_day(d: date) -> bool:
    return d in _HALF_DAYS


def is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:  # Sat/Sun
        return False
    return not is_holiday(d)


def session_info(d: date) -> SessionInfo | None:
    if not is_trading_day(d):
        return SessionInfo(d, datetime.min, datetime.min, False, True)
    half = is_half_day(d)
    open_ny = datetime.combine(d, RTH_OPEN, tzinfo=NY_TZ)
    close_t = HALF_DAY_CLOSE if half else RTH_CLOSE
    close_ny = datetime.combine(d, close_t, tzinfo=NY_TZ)
    return SessionInfo(d, open_ny, close_ny, half, False)


def in_rth(dt: datetime) -> bool:
    """True iff dt (any tz) is inside RTH for its NY-local date, accounting for half-days."""
    ny = to_ny(dt)
    info = session_info(ny.date())
    if info is None or info.is_holiday:
        return False
    return info.open_dt_ny <= ny < info.close_dt_ny


def force_flat_time(d: date) -> time:
    return EOD_FLAT_HALF if is_half_day(d) else EOD_FLAT


def trading_days(start: date, end: date) -> list[date]:
    """Inclusive list of trading days between start and end."""
    out: list[date] = []
    d = start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def opening_range_window(d: date) -> tuple[datetime, datetime]:
    """[start, end) of the 5-min opening range candle in NY tz."""
    return (
        datetime.combine(d, OR_START, tzinfo=NY_TZ),
        datetime.combine(d, OR_END, tzinfo=NY_TZ),
    )
