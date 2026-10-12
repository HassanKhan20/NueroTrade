"""Curated S&P 100-ish universe for cross-sectional momentum.

This is ~100 large-cap US equities chosen for:
  - cross-sector diversity (tech, financials, energy, healthcare, staples,
    industrials, consumer disc, real estate, utilities, materials, comms)
  - high enough liquidity to trade frictionlessly
  - listed across the entire 2022-2026 window (no recent IPOs / failed names)

The list is hand-curated rather than scraped at runtime to keep the backtest
fully reproducible.
"""

SP100_UNIVERSE = [
    # Tech / mega-cap (already in the original 30)
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",
    "ADBE", "CRM", "ORCL", "AMD", "QCOM", "NFLX", "INTC", "CSCO",
    "IBM", "TXN", "ACN", "AMAT", "MU", "ADI", "NOW", "PANW",
    # Communication / media
    "DIS", "CMCSA", "T", "VZ", "TMUS",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "SCHW", "BX",
    "MMC", "CB", "PGR", "USB", "PNC", "TFC", "BK",
    # Healthcare
    "UNH", "LLY", "JNJ", "ABBV", "PFE", "MRK", "TMO", "DHR", "ABT", "BMY",
    "AMGN", "CVS", "GILD", "MDT", "ELV", "ISRG", "VRTX", "REGN", "SYK",
    # Consumer discretionary
    "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "F", "GM", "MAR",
    # Consumer staples
    "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "MDLZ", "CL",
    # Industrials
    "HON", "CAT", "UPS", "RTX", "BA", "GE", "DE", "LMT", "UNP", "ETN",
    # Energy
    "XOM", "CVX", "COP", "OXY", "SLB", "EOG",
    # Materials / commodities
    "LIN", "FCX", "NEM",
    # Utilities
    "NEE", "DUK", "SO",
    # Real estate
    "PLD", "AMT", "EQIX",
    # Payments / fintech (overlap with financials)
    "V", "MA", "PYPL",
]

assert len(set(SP100_UNIVERSE)) == len(SP100_UNIVERSE), "duplicates in SP100"
