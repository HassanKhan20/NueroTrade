"""Biotech / pharma small-mid cap universe for catalyst-momentum.

These are FDA-event-sensitive names tradeable on Alpaca — the types of
stocks that move 30-60% on a single approval, rejection, or trial readout.
Most are $5-100/share so they aren't true penny stocks but they share the
same "catalyst-driven" personality.

Curated for:
  - Listed on NYSE / NASDAQ (Alpaca-tradeable)
  - Avg dollar volume > $5M/day (avoids untradeable micro-caps)
  - Mix of large (REGN, VRTX), mid (BIIB, MRNA), and small biotech (INO, OCGN)
  - Sector skew: oncology, vaccines, gene therapy, generics
"""

BIOTECH_UNIVERSE = [
    # Large-cap biotech (stable benchmarks)
    "REGN", "VRTX", "BIIB", "GILD", "AMGN",
    "ILMN", "ALNY", "BMRN", "INCY", "EXEL",
    # Vaccines / mRNA / covid-era
    "MRNA", "BNTX", "NVAX",
    # Mid-cap specialty pharma & biotech
    "JAZZ", "HALO", "NBIX", "IONS", "ACAD",
    "UTHR", "LEGN", "CRSP", "BLUE", "EDIT",
    "BEAM", "NTLA", "VIR", "ARWR", "FOLD",
    "HRMY", "PCRX", "RNA",
    # Smaller / more speculative biotech (high catalyst beta)
    "INO", "OCGN", "SAVA", "BNGO", "MRTX",
    "VKTX", "VTYX", "ABUS", "SRPT",
    "RXRX", "RVMD", "DNLI",
    # Specialty pharma / generics
    "TEVA", "PRGO", "ENDP", "MNK",
    # Infrastructure / contract research
    "ICLR", "CRL", "DXCM",
]

assert len(set(BIOTECH_UNIVERSE)) == len(BIOTECH_UNIVERSE), "duplicates"
