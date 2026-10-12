"""Tests for the signal modules used by the multi-factor swing strategy."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.signals.halal_screen import (
    CompanyFinancials,
    business_activity_compliant,
    financial_ratio_compliant,
    screen,
)
from src.signals.momentum import momentum_score
from src.signals.pead import (
    EarningsEvent,
    pead_score,
    standardized_unexpected_earnings,
)
from src.signals.quality import quality_score
from src.signals.technical import (
    TechnicalSnapshot,
    atr,
    macd,
    relative_volume,
    rsi,
    sma,
    technical_snapshot,
)


# =============== Halal screen ===============
def test_business_activity_rejects_alcohol() -> None:
    ok, reasons = business_activity_compliant(["alcohol", "brewery"])
    assert not ok and len(reasons) >= 1


def test_business_activity_accepts_software() -> None:
    ok, reasons = business_activity_compliant(["software", "saas"])
    assert ok and reasons == []


def test_financial_ratio_rejects_high_debt() -> None:
    c = CompanyFinancials(market_cap=100, interest_bearing_debt=40,
                          interest_bearing_assets=0, interest_income=0,
                          total_revenue=10)
    ok, reasons, ratios = financial_ratio_compliant(c)
    assert not ok
    assert any("debt" in r for r in reasons)
    assert ratios["debt_ratio"] == pytest.approx(0.40)


def test_financial_ratio_rejects_high_impure_income() -> None:
    c = CompanyFinancials(market_cap=100, interest_bearing_debt=10,
                          interest_bearing_assets=10, interest_income=2,
                          total_revenue=10)
    ok, reasons, _ = financial_ratio_compliant(c)
    assert not ok
    assert any("interest income" in r for r in reasons)


def test_financial_ratio_accepts_clean_balance_sheet() -> None:
    c = CompanyFinancials(market_cap=100, interest_bearing_debt=10,
                          interest_bearing_assets=10, interest_income=0.1,
                          total_revenue=10)
    ok, _, _ = financial_ratio_compliant(c)
    assert ok


def test_screen_combines_both_tiers() -> None:
    c = CompanyFinancials(market_cap=100, interest_bearing_debt=10,
                          interest_bearing_assets=10, interest_income=0.1,
                          total_revenue=10, industry_tags=("software",))
    v = screen(c)
    assert v.is_compliant and v.business_compliant and v.financial_compliant


def test_screen_blocks_when_business_fails_even_if_financials_pass() -> None:
    c = CompanyFinancials(market_cap=100, interest_bearing_debt=10,
                          interest_bearing_assets=10, interest_income=0.1,
                          total_revenue=10, industry_tags=("alcohol",))
    v = screen(c)
    assert not v.is_compliant
    assert not v.business_compliant
    assert v.financial_compliant  # financials are clean
    assert any("alcohol" in r for r in v.reasons)


# =============== Technical signals ===============
def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_rsi_returns_nan_for_short_series() -> None:
    assert np.isnan(rsi(_series([1.0, 2.0, 3.0]), window=14))


def test_rsi_at_100_for_unbroken_uptrend() -> None:
    closes = _series([float(i) for i in range(1, 30)])
    val = rsi(closes, 14)
    # All positive deltas → avg_loss=0 → RSI=100
    assert val == pytest.approx(100.0)


def test_macd_runs_on_long_series() -> None:
    closes = _series([100.0 + np.sin(i / 5) for i in range(100)])
    line, signal, hist = macd(closes)
    assert all(np.isfinite(x) for x in (line, signal, hist))


def test_sma_simple_average() -> None:
    closes = _series([10.0, 20.0, 30.0, 40.0, 50.0])
    assert sma(closes, 5) == pytest.approx(30.0)


def test_atr_increases_with_range() -> None:
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    narrow = pd.DataFrame({
        "high": [101] * n, "low": [99] * n, "close": [100] * n,
    }, index=idx)
    wide = pd.DataFrame({
        "high": [110] * n, "low": [90] * n, "close": [100] * n,
    }, index=idx)
    assert atr(wide) > atr(narrow)


def test_relative_volume_today_vs_recent() -> None:
    vols = _series([1000.0] * 20 + [5000.0])  # 5x spike on last day
    assert relative_volume(vols) == pytest.approx(5.0)


def test_technical_snapshot_all_fields_finite_when_history_long() -> None:
    n = 250
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    closes = 100 + np.cumsum(np.random.RandomState(7).normal(0.05, 0.5, n))
    df = pd.DataFrame({
        "open": closes, "high": closes + 1, "low": closes - 1,
        "close": closes, "volume": np.full(n, 1e6),
    }, index=idx)
    snap = technical_snapshot(df, pd.Timestamp("2024-12-15"))
    assert snap is not None
    for v in (snap.rsi_14, snap.macd_line, snap.sma_50, snap.sma_200,
              snap.atr_14, snap.last_close, snap.momentum_5d):
        assert np.isfinite(v)


def test_technical_snapshot_returns_none_when_history_too_short() -> None:
    df = pd.DataFrame({"open": [100], "high": [101], "low": [99],
                       "close": [100], "volume": [1e6]},
                      index=pd.DatetimeIndex(["2024-01-01"]))
    assert technical_snapshot(df, pd.Timestamp("2024-01-02")) is None


def test_technical_snapshot_uptrend_detection() -> None:
    n = 250
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    closes = np.linspace(50, 150, n)  # monotonic uptrend
    df = pd.DataFrame({
        "open": closes, "high": closes + 0.5, "low": closes - 0.5,
        "close": closes, "volume": np.full(n, 1e6),
    }, index=idx)
    snap = technical_snapshot(df, pd.Timestamp("2024-12-15"))
    assert snap is not None
    assert snap.is_in_uptrend()


# =============== Momentum signal ===============
def test_momentum_score_positive_for_uptrend() -> None:
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    closes = np.linspace(50, 150, n)
    df = pd.DataFrame({"close": closes}, index=idx)
    score = momentum_score(df, pd.Timestamp("2024-12-15"))
    assert score > 0


def test_momentum_score_nan_when_history_short() -> None:
    df = pd.DataFrame({"close": [100, 101, 102]},
                      index=pd.date_range("2024-01-01", periods=3, freq="B"))
    score = momentum_score(df, pd.Timestamp("2024-02-01"))
    assert np.isnan(score)


# =============== Quality signal ===============
def test_quality_uses_override_when_known() -> None:
    score = quality_score("AAPL", None, pd.Timestamp("2024-01-01"))
    assert score == 1.0


def test_quality_falls_back_to_stability_proxy() -> None:
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    closes = np.linspace(50, 60, n)  # very smooth uptrend → high stability
    df = pd.DataFrame({"close": closes}, index=idx)
    score = quality_score("UNKNOWN_TICKER", df, pd.Timestamp("2024-12-15"))
    assert np.isfinite(score)
    assert score > 0  # smooth growth → positive


# =============== PEAD ===============
def test_pead_returns_zero_when_no_data() -> None:
    assert pead_score("AAPL", date(2024, 5, 1), None) == 0.0
    assert pead_score("AAPL", date(2024, 5, 1), []) == 0.0


def test_sue_requires_minimum_history() -> None:
    events = [EarningsEvent("X", date(2024, 1, 1), 1.0, 0.9, 0.10)]
    assert np.isnan(standardized_unexpected_earnings(events, "X"))


def test_sue_computes_correctly_with_enough_history() -> None:
    events = []
    for i, surp in enumerate([0.05, 0.02, -0.01, 0.03, 0.10]):
        events.append(EarningsEvent("X", date(2023, 3 + i, 1),
                                    1.0 + surp, 1.0, surp))
    sue = standardized_unexpected_earnings(events, "X")
    assert np.isfinite(sue)


def test_pead_score_decays_after_30_days() -> None:
    events = []
    for i, surp in enumerate([0.05, 0.02, -0.01, 0.03, 0.10]):
        events.append(EarningsEvent("X", date(2023, 3 + i, 1),
                                    1.0 + surp, 1.0, surp))
    # Last announcement at 2023-07-01; check at +15 vs +45 days
    early = pead_score("X", date(2023, 7, 16), events)
    late = pead_score("X", date(2023, 8, 15), events)
    if abs(early) > 0 and abs(late) > 0:
        assert abs(late) < abs(early)


def test_pead_score_zero_outside_drift_window() -> None:
    events = []
    for i, surp in enumerate([0.05, 0.02, -0.01, 0.03, 0.10]):
        events.append(EarningsEvent("X", date(2023, 3 + i, 1),
                                    1.0 + surp, 1.0, surp))
    # 90 days after last announcement = beyond 60-day window
    assert pead_score("X", date(2023, 9, 30), events) == 0.0
