"""Tests for live price-level event detection."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.signals.level_events import (
    ActiveCallout,
    detect_level_events,
    load_active_callouts,
)


def _active(symbol="X", entry=100.0, stop=95.0, tp1=110.0, tp2=120.0,
            fired=None):
    return ActiveCallout(
        symbol=symbol, grade="B", entry_ts="2026-05-13T00:00:00+00:00",
        entry_price=entry, stop_price=stop,
        take_profit_1=tp1, take_profit_2=tp2,
        cap_tier="mega", halal=True,
        fired=fired or set(),
    )


NOW = datetime(2026, 5, 14, tzinfo=timezone.utc)


def test_tp1_hit_when_price_crosses_target():
    active = {"X": _active()}
    events = detect_level_events(active, {"X": 111.0}, now=NOW)
    assert len(events) == 1
    assert events[0]["event"] == "TP1_HIT"
    assert events[0]["return_pct"] > 0
    # Should not double-emit on same call
    events2 = detect_level_events(active, {"X": 112.0}, now=NOW)
    assert events2 == []


def test_tp2_hit_emits_both_tp1_and_tp2():
    """If price gaps directly above TP2 without TP1 having fired before,
    both events fire — partial fills are honest history."""
    active = {"X": _active()}
    events = detect_level_events(active, {"X": 125.0}, now=NOW)
    kinds = [e["event"] for e in events]
    assert "TP1_HIT" in kinds
    assert "TP2_HIT" in kinds


def test_tp2_alone_when_tp1_already_fired():
    active = {"X": _active(fired={"TP1_HIT"})}
    events = detect_level_events(active, {"X": 121.0}, now=NOW)
    kinds = [e["event"] for e in events]
    assert kinds == ["TP2_HIT"]


def test_stop_hit_when_price_crosses_stop():
    active = {"X": _active()}
    events = detect_level_events(active, {"X": 94.0}, now=NOW)
    assert len(events) == 1
    assert events[0]["event"] == "STOP_HIT"
    assert events[0]["return_pct"] < 0


def test_no_event_when_price_inside_band():
    active = {"X": _active()}
    events = detect_level_events(active, {"X": 105.0}, now=NOW)
    assert events == []


def test_missing_price_silent_skip():
    active = {"X": _active(), "Y": _active(symbol="Y")}
    events = detect_level_events(active, {"X": 111.0}, now=NOW)
    syms = {e["symbol"] for e in events}
    assert syms == {"X"}


def test_already_closed_callout_not_emitted_again():
    active = {"X": _active(fired={"STOP_HIT"})}
    events = detect_level_events(active, {"X": 50.0}, now=NOW)
    assert events == []
    events = detect_level_events(active, {"X": 130.0}, now=NOW)
    assert events == []


def test_load_active_excludes_closed_callouts(tmp_path):
    feed = tmp_path / "feed.jsonl"
    with feed.open("w", encoding="utf-8") as f:
        # MU: NEW then STOP_HIT — should NOT appear in active
        f.write(json.dumps({
            "ts": "2026-05-13T00:00:00+00:00", "event": "NEW", "symbol": "MU",
            "grade": "C", "entry_price": 500, "stop_price": 475,
            "take_profit_1": 525, "take_profit_2": 550,
            "cap_tier": "mega", "halal": True,
        }) + "\n")
        f.write(json.dumps({
            "ts": "2026-05-13T05:00:00+00:00", "event": "STOP_HIT",
            "symbol": "MU", "grade": "C", "entry_price": 500,
            "stop_price": 475, "take_profit_1": 525, "take_profit_2": 550,
            "current_price": 470, "cap_tier": "mega", "halal": True,
        }) + "\n")
        # AAPL: NEW only — should appear in active
        f.write(json.dumps({
            "ts": "2026-05-13T01:00:00+00:00", "event": "NEW",
            "symbol": "AAPL", "grade": "B", "entry_price": 200,
            "stop_price": 190, "take_profit_1": 215, "take_profit_2": 230,
            "cap_tier": "mega", "halal": True,
        }) + "\n")
    active = load_active_callouts(feed)
    assert "MU" not in active
    assert "AAPL" in active


def test_load_active_keeps_open_callout_after_tp1():
    """TP1 doesn't close the trade — runner stays on for TP2."""
    p = Path("/tmp/feed_test_tp1.jsonl")  # use tmp_path in real test
    # build inline for test
    events = [
        {"ts": "2026-05-13T00:00:00+00:00", "event": "NEW", "symbol": "X",
         "grade": "B", "entry_price": 100, "stop_price": 95,
         "take_profit_1": 110, "take_profit_2": 120,
         "cap_tier": "mega", "halal": True},
        {"ts": "2026-05-13T03:00:00+00:00", "event": "TP1_HIT", "symbol": "X",
         "grade": "B", "entry_price": 100, "stop_price": 95,
         "take_profit_1": 110, "take_profit_2": 120,
         "current_price": 110, "cap_tier": "mega", "halal": True},
    ]
    p.write_text("\n".join(json.dumps(e) for e in events))
    active = load_active_callouts(p)
    assert "X" in active
    assert "TP1_HIT" in active["X"].fired
