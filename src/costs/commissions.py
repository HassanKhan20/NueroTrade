"""Per-share commission model.

Default uses the Zarattini-Aziz (SSRN 2024) convention of $0.0035/share with
a $0.35 minimum per ticket. Even though Alpaca itself charges $0, we model a
non-zero commission as a slippage buffer in backtests.
"""
from __future__ import annotations

DEFAULT_PER_SHARE = 0.0035
DEFAULT_MIN_TICKET = 0.35


def commission(
    shares: float,
    *,
    per_share: float = DEFAULT_PER_SHARE,
    min_ticket: float = DEFAULT_MIN_TICKET,
) -> float:
    """Return absolute commission for a single fill.

    Always returns a non-negative number. ``shares`` is treated as absolute
    (i.e. sign-agnostic — a 100-share buy and 100-share sell cost the same).
    """
    if shares == 0:
        return 0.0
    raw = abs(shares) * per_share
    return max(raw, min_ticket)


def round_trip_cost(
    shares: float,
    *,
    per_share: float = DEFAULT_PER_SHARE,
    min_ticket: float = DEFAULT_MIN_TICKET,
) -> float:
    """Entry + exit commission combined."""
    return 2.0 * commission(shares, per_share=per_share, min_ticket=min_ticket)
