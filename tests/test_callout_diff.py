"""Tests for callout-feed event diffing."""
from __future__ import annotations

from datetime import datetime, timezone

from src.signals.callout_diff import diff_picks


def _pick(sym: str, grade: str, action: str = "ENTER — full size") -> dict:
    return {
        "symbol": sym, "grade": grade, "composite_score": 1.5,
        "entry_price": 100.0, "stop_price": 95.0,
        "take_profit_1": 110.0, "take_profit_2": 120.0,
        "reward_to_risk_tp1": 2.0, "reward_to_risk_tp2": 4.0,
        "suggested_shares": 100, "notional": 10000.0,
        "action": action, "atr_14": 2.5, "rank": 1,
    }


T0 = datetime(2026, 5, 7, 14, 0, tzinfo=timezone.utc)


def test_new_pick_emits_new_event():
    events = diff_picks(prev=[], curr=[_pick("AAPL", "B")], now=T0)
    assert len(events) == 1
    assert events[0]["event"] == "NEW"
    assert events[0]["symbol"] == "AAPL"


def test_upgraded_grade_emits_upgraded():
    prev = [_pick("MSFT", "C")]
    curr = [_pick("MSFT", "A")]
    events = diff_picks(prev=prev, curr=curr, now=T0)
    assert len(events) == 1
    assert events[0]["event"] == "UPGRADED"
    assert events[0]["prev_grade"] == "C"
    assert events[0]["grade"] == "A"


def test_downgraded_grade_emits_downgraded():
    prev = [_pick("NVDA", "A")]
    curr = [_pick("NVDA", "C")]
    events = diff_picks(prev=prev, curr=curr, now=T0)
    assert len(events) == 1
    assert events[0]["event"] == "DOWNGRADED"
    assert events[0]["prev_grade"] == "A"


def test_dropped_emits_dropped():
    prev = [_pick("META", "B")]
    curr = []
    events = diff_picks(prev=prev, curr=curr, now=T0)
    assert len(events) == 1
    assert events[0]["event"] == "DROPPED"
    assert events[0]["symbol"] == "META"


def test_unchanged_no_event():
    p = _pick("AMD", "B")
    events = diff_picks(prev=[p], curr=[p], now=T0)
    assert events == []


def test_action_change_emits_event():
    prev = [_pick("KLAC", "B", action="WAIT — pullback")]
    curr = [_pick("KLAC", "B", action="ENTER — full size")]
    events = diff_picks(prev=prev, curr=curr, now=T0)
    assert len(events) == 1
    assert events[0]["event"] == "ACTION_CHANGED"
    assert events[0]["prev_action"] == "WAIT — pullback"
    assert events[0]["action"] == "ENTER — full size"


def test_below_floor_grade_ignored():
    # F-grade churn should not pollute the feed
    prev = [_pick("WBA", "F")]
    curr = [_pick("WBA", "D")]
    events = diff_picks(prev=prev, curr=curr, now=T0)
    assert events == []


def test_upgrade_supersedes_action_change():
    # If grade moves AND action verb changes, only one event (UPGRADED) emits
    prev = [_pick("INTC", "C", action="REDUCE")]
    curr = [_pick("INTC", "A", action="ENTER")]
    events = diff_picks(prev=prev, curr=curr, now=T0)
    syms = [e["symbol"] for e in events]
    assert syms.count("INTC") == 1
    assert events[0]["event"] == "UPGRADED"
