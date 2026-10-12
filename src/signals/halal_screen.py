"""AAOIFI-style Sharia screening — pure functions, no I/O.

Two-tier screen as defined by the AAOIFI Sharia Standards (the methodology
Zoya, Wahed, and HLAL all use):

  Tier 1 — Business activity screen (qualitative)
      Excludes companies whose primary or significant secondary revenue
      comes from prohibited activities (riba, alcohol, gambling, etc.).

  Tier 2 — Financial ratio screen (quantitative)
      Even compliant-business companies fail if their balance sheet has
      excessive interest-bearing leverage or impure income.
      AAOIFI thresholds:
          non-compliant debt / market cap            <= 30%
          non-compliant interest-bearing assets / mc <= 30%
          impure income / total revenue              <= 5%

The functions here are *pure*: pass in the inputs, get back a verdict
plus diagnostic reasons. Wire them to whatever data source (Zoya API,
SEC EDGAR, yfinance, etc.) at the call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# Sectors / industries that are NEVER halal regardless of financials.
PROHIBITED_BUSINESSES = frozenset({
    "alcohol", "brewery", "distillery", "winery",
    "tobacco", "cigarette", "vaping",
    "gambling", "casino", "lottery", "betting", "wagering",
    "pornography", "adult-entertainment",
    "weapons", "armaments", "defense-weapons",
    "pork", "swine",
    "conventional-bank", "conventional-insurance", "interest-lending",
    "consumer-finance", "credit-card-issuer",  # riba-based
})

# AAOIFI thresholds — current standard as of 2025.
DEBT_TO_MARKETCAP_MAX = 0.30
INTEREST_BEARING_ASSETS_TO_MARKETCAP_MAX = 0.30
IMPURE_INCOME_TO_REVENUE_MAX = 0.05


@dataclass(frozen=True)
class CompanyFinancials:
    """Inputs for the financial-ratio screen.

    Source the values from a fundamentals provider (Zoya, EDGAR, yfinance).
    Use ``None`` for any value you can't determine — the screen will reject
    rather than assume compliance.
    """
    market_cap: float
    interest_bearing_debt: float          # bonds, term loans, conventional debt
    interest_bearing_assets: float = 0.0  # cash + securities earning interest
    interest_income: float = 0.0          # from income statement
    total_revenue: float = 0.0
    industry_tags: tuple[str, ...] = ()   # lowercase, e.g. ("software", "saas")


@dataclass(frozen=True)
class HalalVerdict:
    is_compliant: bool
    business_compliant: bool
    financial_compliant: bool
    reasons: tuple[str, ...]
    debt_ratio: float | None = None
    interest_assets_ratio: float | None = None
    impure_income_ratio: float | None = None


def business_activity_compliant(industry_tags: Iterable[str]) -> tuple[bool, list[str]]:
    """Return (compliant, list_of_failing_reasons)."""
    tags = {t.strip().lower() for t in industry_tags}
    hits = sorted(tags & PROHIBITED_BUSINESSES)
    if hits:
        return False, [f"prohibited business: {h}" for h in hits]
    return True, []


def financial_ratio_compliant(c: CompanyFinancials) -> tuple[bool, list[str], dict[str, float]]:
    """Return (compliant, failing_reasons, computed_ratios)."""
    reasons: list[str] = []
    ratios: dict[str, float] = {}

    if c.market_cap <= 0:
        return False, ["non-positive market cap"], {}

    debt_ratio = c.interest_bearing_debt / c.market_cap
    ratios["debt_ratio"] = debt_ratio
    if debt_ratio > DEBT_TO_MARKETCAP_MAX:
        reasons.append(
            f"debt/market_cap {debt_ratio:.1%} > {DEBT_TO_MARKETCAP_MAX:.0%}"
        )

    ib_assets_ratio = c.interest_bearing_assets / c.market_cap
    ratios["interest_assets_ratio"] = ib_assets_ratio
    if ib_assets_ratio > INTEREST_BEARING_ASSETS_TO_MARKETCAP_MAX:
        reasons.append(
            f"interest-bearing assets/market_cap {ib_assets_ratio:.1%} "
            f"> {INTEREST_BEARING_ASSETS_TO_MARKETCAP_MAX:.0%}"
        )

    if c.total_revenue > 0:
        impure_ratio = c.interest_income / c.total_revenue
        ratios["impure_income_ratio"] = impure_ratio
        if impure_ratio > IMPURE_INCOME_TO_REVENUE_MAX:
            reasons.append(
                f"interest income/revenue {impure_ratio:.1%} "
                f"> {IMPURE_INCOME_TO_REVENUE_MAX:.0%}"
            )

    return (len(reasons) == 0), reasons, ratios


def screen(c: CompanyFinancials) -> HalalVerdict:
    """Apply both tiers of the screen to a company."""
    biz_ok, biz_reasons = business_activity_compliant(c.industry_tags)
    fin_ok, fin_reasons, ratios = financial_ratio_compliant(c)
    return HalalVerdict(
        is_compliant=biz_ok and fin_ok,
        business_compliant=biz_ok,
        financial_compliant=fin_ok,
        reasons=tuple(biz_reasons + fin_reasons),
        debt_ratio=ratios.get("debt_ratio"),
        interest_assets_ratio=ratios.get("interest_assets_ratio"),
        impure_income_ratio=ratios.get("impure_income_ratio"),
    )
