"""Tests for the self-grading outcomes scorer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.signals.callout_outcomes import (
    aggregate,
    resolve_outcome,
)


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """rows: [(date, open, high, low, close), ...]"""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    return pd.DataFrame({
        "open":  [r[1] for r in rows],
        "high":  [r[2] for r in rows],
        "low":   [r[3] for r in rows],
        "close": [r[4] for r in rows],
        "volume": [1_000_000] * len(rows),
    }, index=idx)


def _ev(**kw):
    base = {
        "event": "NEW", "symbol": "X", "grade": "B",
        "ts": "2026-04-01T12:00:00+00:00",
        "entry_price": 100.0, "stop_price": 95.0,
        "take_profit_1": 110.0, "take_profit_2": 120.0,
        "cap_tier": "mega", "halal": True,
    }
    base.update(kw)
    return base


NOW = datetime(2026, 5, 1, tzinfo=timezone.utc)


def test_hit_tp1_when_high_crosses_tp1():
    bars = _bars([
        ("2026-04-02", 100, 105, 99, 104),
        ("2026-04-03", 104, 111, 103, 110),  # high 111 -> TP1 hit
    ])
    o = resolve_outcome(_ev(), bars, now=NOW)
    assert o.outcome == "HIT_TP1"
    assert o.return_pct is not None and o.return_pct > 0


def test_hit_tp2_when_high_crosses_tp2():
    bars = _bars([
        ("2026-04-02", 100, 121, 99, 120),  # blew past TP2
    ])
    o = resolve_outcome(_ev(), bars, now=NOW)
    assert o.outcome == "HIT_TP2"


def test_hit_stop_when_low_crosses_stop():
    bars = _bars([
        ("2026-04-02", 100, 102, 94, 96),  # low 94 < stop 95
    ])
    o = resolve_outcome(_ev(), bars, now=NOW)
    assert o.outcome == "HIT_STOP"


def test_same_day_stop_and_tp1_assumes_stop_first():
    # Conservative: if both bounds touched same bar, stop wins.
    bars = _bars([
        ("2026-04-02", 100, 112, 94, 100),
    ])
    o = resolve_outcome(_ev(), bars, now=NOW)
    assert o.outcome == "HIT_STOP"


def test_open_when_no_bound_touched():
    bars = _bars([
        ("2026-04-02", 100, 102, 99, 101),
        ("2026-04-03", 101, 103, 100, 102),
    ])
    # Within window, neither hit -> OPEN
    o = resolve_outcome(_ev(), bars, now=datetime(2026, 4, 4, tzinfo=timezone.utc))
    assert o.outcome == "OPEN"


def test_expired_after_horizon_no_hit():
    # 22 bars all hovering, horizon_days=21 => EXPIRED
    rows = [(f"2026-04-{i+2:02d}", 100, 102, 99, 101) for i in range(22)]
    bars = _bars(rows)
    o = resolve_outcome(_ev(), bars, horizon_days=21,
                         now=datetime(2026, 5, 5, tzinfo=timezone.utc))
    assert o.outcome == "EXPIRED"


def test_skipped_when_entry_missing():
    bars = _bars([("2026-04-02", 100, 102, 99, 101)])
    o = resolve_outcome(_ev(entry_price=0), bars, now=NOW)
    assert o.outcome == "SKIPPED"


def test_aggregate_counts_buckets():
    # 4 outcomes: 2 TP1 wins, 1 stop, 1 open
    from src.signals.callout_outcomes import CalloutOutcome
    outs = [
        CalloutOutcome("A", "B", "mega", True, "NEW", "t", 5.0,
                        100, 95, 110, 120, "HIT_TP1", 110, 10.0),
        CalloutOutcome("B", "A", "mid",  True, "NEW", "t", 5.0,
                        50, 47, 55, 60, "HIT_TP1", 55, 10.0),
        CalloutOutcome("C", "B", "mid", False, "NEW", "t", 5.0,
                        20, 19, 22, 24, "HIT_STOP", 19, -5.0),
        CalloutOutcome("D", "C", "mega", False, "NEW", "t", 1.0,
                        10, 9, 11, 12, "OPEN", 10.5, 5.0),
    ]
    a = aggregate(outs)
    assert a.overall.n == 4
    assert a.overall.n_resolved == 3
    assert a.overall.n_open == 1
    assert a.overall.n_tp1 == 2
    assert a.overall.n_stop == 1
    assert abs(a.overall.hit_rate - 2/3) < 1e-9
