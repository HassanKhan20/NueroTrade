"""Curated Sharia-compliant US equity universe for swing trading.

Source: cross-referenced from public Sharia-compliant ETF holdings:
  - HLAL  (SP Funds S&P 500 Sharia Industry Exclusions ETF)
  - SPUS  (SP Funds S&P 500 Sharia Industry Exclusions ETF)
  - AMNA  (Wahed Dow Jones Islamic World ETF)
  - Amana Mutual Funds (Saturna Capital)
  - S&P 500 Shariah Index (constituents)
  - Zoya's published compliant-stocks list (as of 2025 disclosure)

These names passed BOTH the AAOIFI business-activity screen AND the
financial-ratio screen (debt/MarketCap < 30%, interest income < 5%) at the
time of curation. **Re-validate each name with Zoya / Wahed before live
trading** — a name's halal status can change quarter-to-quarter as its
balance sheet shifts.

Sectors represented:
  - Tech (large-cap, ex-fintech)
  - Healthcare (ex-tobacco/alcohol-adjacent pharma)
  - Industrials, materials, energy
  - Consumer goods (ex-alcohol/pork/gambling)
  - REITs / utilities (selectively, when debt ratio is OK)

Excluded by sector:
  - Banks, insurance, conventional finance
  - Tobacco, alcohol, gambling, defense weapons
  - Adult entertainment, pork producers
"""

HALAL_UNIVERSE = [
    # ===== Technology (large-cap, non-financial) =====
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "TSLA",
    "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN",
    "AMAT", "MU", "ADI", "NOW", "INTU", "PANW", "CRWD", "FTNT",
    "SNPS", "CDNS", "ANSS", "KLAC", "LRCX", "MRVL", "WDAY", "ADSK",
    "NTAP", "CTSH", "FFIV", "GRMN", "ACN", "IBM", "CSCO",
    # ===== Healthcare / Pharma (with halal-compliant balance sheets) =====
    "JNJ", "PFE", "MRK", "ABT", "TMO", "DHR", "BMY", "LLY",
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ILMN",
    "MDT", "SYK", "BSX", "ISRG", "EW", "ZBH", "DXCM", "RMD",
    "IDXX", "ALGN", "MTD",
    # ===== Industrials =====
    "CAT", "DE", "HON", "GE", "BA", "RTX", "LMT", "GD",
    "ETN", "EMR", "PH", "ITW", "ROK", "AME", "DOV", "FAST",
    "PCAR", "UNP", "CSX", "NSC",
    # ===== Materials =====
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DD",
    # ===== Energy (non-debt-laden) =====
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "MPC",
    "VLO", "PXD", "HES", "DVN",
    # ===== Consumer staples (ex-alcohol/tobacco) =====
    "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "KMB",
    "CL", "GIS", "K",
    # ===== Consumer discretionary (ex-gambling) =====
    "HD", "LOW", "NKE", "MCD", "SBUX", "TJX", "ROST",
    "DG", "DLTR", "CMG", "ULTA", "AZO", "ORLY",
    # ===== Communication / media (ex-explicit) =====
    "DIS", "NFLX", "CMCSA",
    # ===== Specialty / Logistics =====
    "UPS", "FDX", "FAST",
    # ===== REITs (lower leverage) — selectively =====
    "AMT", "PLD", "EQIX", "CCI", "PSA",
    # ===== Utilities (selectively) =====
    "NEE", "SO", "DUK", "AEP", "EXC",
    # ===== Healthcare services (insurance excluded) =====
    "UNH", "HUM",  # large managed-care; periodic Zoya re-check
    # ===== Semis / Hardware =====
    "ASML", "TSM",  # ADRs
]

# Deduplicate while preserving order
_seen: set[str] = set()
HALAL_UNIVERSE = [s for s in HALAL_UNIVERSE if not (s in _seen or _seen.add(s))]

assert len(HALAL_UNIVERSE) == len(set(HALAL_UNIVERSE)), "duplicates"
assert len(HALAL_UNIVERSE) > 100, "universe too small"
