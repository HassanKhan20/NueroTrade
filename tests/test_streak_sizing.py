"""Tests for win-streak (anti-martingale) sizing."""
from __future__ import annotations

import json
from pathlib import Path

from src.signals.streak_sizing import (
    CAP_PCT,
    current_win_streak,
    streak_risk_pct,
)


def test_base_risk_at_zero_streak():
    assert streak_risk_pct(0, base_pct=0.01, step=0.005) == 0.01


def test_risk_grows_with_streak():
    assert streak_risk_pct(1, base_pct=0.01, step=0.005) == 0.015
    assert streak_risk_pct(3, base_pct=0.01, step=0.005) == 0.025


def test_risk_capped():
    # streak of 20 would be way over the cap -> clamped
    assert streak_risk_pct(20, base_pct=0.01, step=0.005, cap_pct=0.03) == 0.03


def test_negative_streak_treated_as_zero():
    assert streak_risk_pct(-5, base_pct=0.01) == 0.01


def _feed(tmp_path: Path, events: list[dict]) -> Path:
    p = tmp_path / "wfeed.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events))
    return p


def _ev(kind, sym, ts, entry_ts=None):
    return {"event": kind, "symbol": sym, "ts": ts,
            "entry_ts": entry_ts or ts}


def test_streak_counts_consecutive_wins(tmp_path):
    feed = _feed(tmp_path, [
        _ev("TP1_HIT", "A", "2026-05-01T00:00:00+00:00"),
        _ev("TP2_HIT", "B", "2026-05-02T00:00:00+00:00"),
        _ev("TP1_HIT", "C", "2026-05-03T00:00:00+00:00"),
    ])
    assert current_win_streak(feed) == 3


def test_streak_resets_on_loss(tmp_path):
    feed = _feed(tmp_path, [
        _ev("TP1_HIT", "A", "2026-05-01T00:00:00+00:00"),
        _ev("STOP_HIT", "B", "2026-05-02T00:00:00+00:00"),  # most recent = loss
        _ev("TP1_HIT", "C", "2026-05-03T00:00:00+00:00"),   # newest = win
    ])
    # Newest is C (win) then B (loss) -> streak = 1
    assert current_win_streak(feed) == 1


def test_streak_zero_when_latest_is_loss(tmp_path):
    feed = _feed(tmp_path, [
        _ev("TP1_HIT", "A", "2026-05-01T00:00:00+00:00"),
        _ev("STOP_HIT", "B", "2026-05-05T00:00:00+00:00"),  # latest = loss
    ])
    assert current_win_streak(feed) == 0


def test_tp1_then_tp2_same_callout_counts_once(tmp_path):
    feed = _feed(tmp_path, [
        _ev("TP1_HIT", "A", "2026-05-01T00:00:00+00:00", entry_ts="e1"),
        _ev("TP2_HIT", "A", "2026-05-02T00:00:00+00:00", entry_ts="e1"),
    ])
    # Same callout (entry_ts e1) -> one win, not two
    assert current_win_streak(feed) == 1


def test_empty_feed_zero_streak(tmp_path):
    feed = _feed(tmp_path, [])
    assert current_win_streak(feed) == 0
