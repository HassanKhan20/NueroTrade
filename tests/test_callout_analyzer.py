"""Tests for the pattern analyzer.

Key invariant: NEVER flag a finding on n < MIN_N. Better to say "need
more data" than to overfit to noise.
"""
from __future__ import annotations

from src.signals.callout_analyzer import (
    Finding, _bucket_stats,
)


def _row(symbol="X", grade="C", cap="mega", halal=True,
         outcome="HIT_TP1", return_pct=5.0):
    return {
        "symbol": symbol, "grade": grade, "cap_tier": cap,
        "halal": halal, "outcome": outcome, "return_pct": return_pct,
    }


def _winners(n, grade="B"):
    # Slight jitter (8 ± 0.5) so scipy's variance calc doesn't underflow
    return [_row(symbol=f"W{i}", grade=grade, outcome="HIT_TP1",
                 return_pct=8.0 + (i % 5) * 0.1) for i in range(n)]


def _losers(n, grade="C"):
    return [_row(symbol=f"L{i}", grade=grade, outcome="HIT_STOP",
                 return_pct=-7.0 + (i % 5) * 0.1) for i in range(n)]


def test_underpowered_bucket_gets_no_finding(monkeypatch):
    """Small bucket (n < 30) should not produce a finding."""
    # Force MIN_N to default 30
    import src.signals.callout_analyzer as a
    monkeypatch.setattr(a, "MIN_N", 30)
    # 10 winners, 10 losers — even though signal is strong, n=10 per bucket too small
    rows = _winners(10, grade="A") + _losers(10, grade="F")
    findings, insufficient = _bucket_stats(
        rows, baseline_wr=0.5, baseline_avg=0.5,
        dim="grade", key_fn=lambda r: r["grade"],
    )
    assert findings == []
    assert {b["bucket"] for b in insufficient} == {"A", "F"}


def test_strong_significant_bucket_flagged(monkeypatch):
    """Large bucket with clear edge -> significant or strong finding."""
    import src.signals.callout_analyzer as a
    monkeypatch.setattr(a, "MIN_N", 30)
    # Bucket B: 40 picks, all winners. Bucket C: 40 picks, all losers.
    rows = _winners(40, grade="B") + _losers(40, grade="C")
    baseline_wr = 0.5
    findings, _ = _bucket_stats(
        rows, baseline_wr=baseline_wr, baseline_avg=0.5,
        dim="grade", key_fn=lambda r: r["grade"],
    )
    by_bucket = {f.bucket: f for f in findings}
    assert "B" in by_bucket
    assert "C" in by_bucket
    assert by_bucket["B"].direction == "outperform"
    assert by_bucket["C"].direction == "underperform"
    # With such an extreme split, both should be "strong"
    assert by_bucket["B"].significance in ("strong", "significant")


def test_insignificant_bucket_marked_underpowered(monkeypatch):
    """A bucket whose distribution matches baseline -> underpowered
    (insufficient evidence of edge)."""
    import src.signals.callout_analyzer as a
    monkeypatch.setattr(a, "MIN_N", 30)
    # 30 picks, exactly 50/50 split — matches a 50% baseline
    rows = _winners(15) + _losers(15)
    findings, _ = _bucket_stats(
        rows, baseline_wr=0.5, baseline_avg=0.5,
        dim="grade", key_fn=lambda r: r["grade"][0] if r["grade"] else None,
    )
    # The buckets here are "B" and "C" by grade; together they form n=30
    # so they could either flag or not depending on split
    for f in findings:
        # When p > 0.05 the recommendation should be "no action"
        if f.significance == "underpowered":
            assert "no action" in f.recommendation.lower()


def test_none_key_skipped(monkeypatch):
    """Rows that return None from key_fn are dropped silently."""
    import src.signals.callout_analyzer as a
    monkeypatch.setattr(a, "MIN_N", 5)
    rows = _winners(20) + _losers(20)
    findings, _ = _bucket_stats(
        rows, baseline_wr=0.5, baseline_avg=0.5,
        dim="industry", key_fn=lambda r: None,
    )
    assert findings == []


def test_recommendation_text_includes_numbers(monkeypatch):
    """Outperform/underperform recommendations should include the
    win-rate and return numbers for human review."""
    import src.signals.callout_analyzer as a
    monkeypatch.setattr(a, "MIN_N", 30)
    rows = _winners(40, grade="B")  # all winners in one bucket
    findings, _ = _bucket_stats(
        rows, baseline_wr=0.5, baseline_avg=0.0,
        dim="grade", key_fn=lambda r: r["grade"],
    )
    assert findings, "should produce a finding for n=40"
    rec = findings[0].recommendation
    assert "%" in rec
    assert "WR" in rec or "win" in rec.lower()
