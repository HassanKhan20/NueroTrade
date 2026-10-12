"""Curated Sharia-compliant US **mid-cap** universe for small-account swing trading.

Why a separate list:
  HALAL_UNIVERSE is dominated by mega-caps (AAPL, KLAC, CAT, etc.) where one
  share costs hundreds-to-thousands of dollars. A trader with $500-$5,000 of
  risk capital cannot meaningfully size positions in those names — at 1% risk
  per trade ($5 on $500) you'd buy zero shares of MU at $518.

  This list targets US mid-caps (roughly $2B-$10B market cap, share prices
  mostly in the $20-$150 range) that ALSO pass the AAOIFI two-tier Sharia
  screen at the time of curation:
    - Business activity is not in {alcohol, tobacco, gambling, conventional
      finance/insurance, weapons, adult, pork}
    - Debt / market cap < ~30%
    - Interest income / revenue < ~5%

  As with HALAL_UNIVERSE, **re-validate each name with Zoya / Wahed before
  putting real capital at risk** — Sharia compliance flips quarter-to-quarter
  as balance sheets shift.

Bias: lean toward profitable mid-caps with low debt rather than unprofitable
SaaS rocketships. The screener's quality factor will already penalise the
former, but starting from a cleaner pool reduces noise.
"""

MIDCAP_UNIVERSE = [
    # ===== Semiconductors & semicap (mid-cap) =====
    "ENTG",   # Entegris
    "AMKR",   # Amkor Technology
    "POWI",   # Power Integrations
    "MKSI",   # MKS Instruments
    "SLAB",   # Silicon Labs
    "CRUS",   # Cirrus Logic
    "LSCC",   # Lattice Semiconductor
    "ONTO",   # Onto Innovation
    "FORM",   # FormFactor
    "COHR",   # Coherent Corp
    "SITM",   # SiTime
    "AEIS",   # Advanced Energy Industries

    # ===== Software / IT services (profitable, low-debt) =====
    "TENB",   # Tenable
    "QLYS",   # Qualys
    "PCTY",   # Paylocity
    "PAYC",   # Paycom
    "DT",     # Dynatrace
    "ESTC",   # Elastic
    "JAMF",   # Jamf
    "BSY",    # Bentley Systems
    "MANH",   # Manhattan Associates
    "PRGS",   # Progress Software
    "ACIW",   # ACI Worldwide
    "ALRM",   # Alarm.com

    # ===== Healthcare / med-devices / life sciences =====
    "BRKR",   # Bruker
    "MASI",   # Masimo
    "ICUI",   # ICU Medical
    "ATR",    # AptarGroup
    "ENSG",   # Ensign Group
    "CHE",    # Chemed
    "HQY",    # HealthEquity
    "AMN",    # AMN Healthcare
    "PRGO",   # Perrigo
    "WST",    # West Pharmaceutical (lower-cap end)
    "NEOG",   # Neogen
    "TFX",    # Teleflex
    "CRL",    # Charles River Labs
    "MEDP",   # Medpace

    # ===== Industrials (capital-light, low-debt) =====
    "NDSN",   # Nordson
    "GGG",    # Graco
    "WTS",    # Watts Water
    "TKR",    # Timken
    "AOS",    # A.O. Smith
    "RBC",    # RBC Bearings
    "SAIA",   # Saia (LTL trucking)
    "ARCB",   # ArcBest
    "RXO",    # RXO logistics
    "KFY",    # Korn Ferry
    "EXPO",   # Exponent
    "MMS",    # Maximus
    "FA",     # First Advantage
    "FELE",   # Franklin Electric
    "WTTR",   # Select Water Solutions

    # ===== Materials / specialty chems (low-debt names only) =====
    "AVNT",   # Avient
    "EXP",    # Eagle Materials
    "SLGN",   # Silgan Holdings
    "CMC",    # Commercial Metals
    "BCC",    # Boise Cascade
    "TREX",   # Trex
    "AZEK",   # Azek
    "SCL",    # Stepan Co
    "GEF",    # Greif
    "MTH",    # Meritage Homes (homebuilder, mostly cash)
    "TPH",    # Tri Pointe Homes

    # ===== Consumer (ex-alcohol/tobacco/gambling) =====
    "MNRO",   # Monro
    "SCVL",   # Shoe Carnival
    "CROX",   # Crocs
    "PLNT",   # Planet Fitness
    "FIZZ",   # National Beverage
    "KTB",    # Kontoor Brands
    "CRI",    # Carter's
    "SHOO",   # Steve Madden
    "BOOT",   # Boot Barn
    "DECK",   # Deckers Outdoor (large-mid)
    "CAKE",   # Cheesecake Factory
    "WING",   # Wingstop
    "TXRH",   # Texas Roadhouse

    # ===== Energy / commodity producers (no leveraged drillers) =====
    "MTDR",   # Matador Resources
    "SM",     # SM Energy
    "MUR",    # Murphy Oil
    "RRC",    # Range Resources
    "PR",     # Permian Resources
    "CIVI",   # Civitas Resources
    "CHRD",   # Chord Energy

    # ===== Tech / internet (smaller-cap, profitable) =====
    "SNAP",   # Snap (verify with Zoya — content filter borderline)
    "DOCN",   # DigitalOcean
    "MNDY",   # monday.com
]
