"""Tests for the industry correlation cap on rank_universe."""
from __future__ import annotations

from src.strategies.multi_factor_swing import apply_industry_cap


def test_cap_drops_excess_in_same_industry(monkeypatch):
    # Simulate 5 names from the same industry plus 2 from others.
    industries = {
        "MU": "Semiconductors", "AMD": "Semiconductors",
        "NVDA": "Semiconductors", "INTC": "Semiconductors",
        "AVGO": "Semiconductors",
        "JPM": "Banks - Diversified",
        "XOM": "Oil & Gas Integrated",
    }
    monkeypatch.setattr(
        "config.universe_meta.industry_for",
        lambda s: industries.get(s),
    )

    ranked = [
        ("MU",   2.0, {}),
        ("AMD",  1.9, {}),
        ("NVDA", 1.8, {}),
        ("INTC", 1.7, {}),
        ("AVGO", 1.6, {}),
        ("JPM",  1.5, {}),
        ("XOM",  1.4, {}),
    ]
    capped = apply_industry_cap(ranked, max_per_industry=2)
    syms = [t[0] for t in capped]
    # Keeps top 2 semis (MU, AMD) and the others
    assert syms == ["MU", "AMD", "JPM", "XOM"]


def test_cap_disabled_returns_original(monkeypatch):
    monkeypatch.setattr(
        "config.universe_meta.industry_for", lambda s: "Anything",
    )
    ranked = [("A", 1.0, {}), ("B", 0.5, {})]
    assert apply_industry_cap(ranked, 0) == ranked


def test_cap_unknown_industry_grouped_under_one_bucket(monkeypatch):
    monkeypatch.setattr(
        "config.universe_meta.industry_for", lambda s: None,
    )
    ranked = [("A", 2, {}), ("B", 1, {}), ("C", 0, {})]
    capped = apply_industry_cap(ranked, max_per_industry=2)
    # All None industries share one bucket; cap=2 keeps the top 2
    assert [t[0] for t in capped] == ["A", "B"]


def test_cap_preserves_score_order_within_industry(monkeypatch):
    monkeypatch.setattr(
        "config.universe_meta.industry_for",
        lambda s: "Semis" if s in ("A", "B", "C") else "Banks",
    )
    ranked = [
        ("A", 3, {}), ("X", 2.5, {}), ("B", 2, {}),
        ("Y", 1.5, {}), ("C", 1, {}),
    ]
    capped = apply_industry_cap(ranked, max_per_industry=2)
    # Industry-cap=2 should keep top 2 semis (A, B) and both banks (X, Y)
    assert [t[0] for t in capped] == ["A", "X", "B", "Y"]
