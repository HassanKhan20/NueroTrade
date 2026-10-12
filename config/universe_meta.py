"""Master symbol catalog — single source of truth for cap-tier and halal status.

Every symbol the screener can rank lives here with two tags:
  - cap_tier : "mega" (~$10B+) or "mid" (~$2B-$10B)
  - halal    : True if the name passes a conservative AAOIFI two-tier screen
               at curation time, False otherwise

The watcher scans the union of all symbols by default. The dashboard tabs
filter the visible callouts using these tags, so users can flip between
"all", "halal only", "non-halal only", "mega caps only", "mid caps only".

Re-validate halal flags with Zoya / Wahed before committing real capital —
balance sheets shift every quarter.
"""
from __future__ import annotations

from typing import Literal, TypedDict

CapTier = Literal["mega", "mid"]


class SymbolMeta(TypedDict):
    cap: CapTier
    halal: bool


# Imports done lazily so tests / scripts can swap pieces.
from config.halal_universe import HALAL_UNIVERSE
from config.midcap_universe import MIDCAP_UNIVERSE


# ============================================================
# Non-halal additions: financials, insurance, traditional
# brokers, asset managers — names that fail the AAOIFI screens
# (interest income > 5%, debt > 30%, or business activity).
# ============================================================
NON_HALAL_MEGACAP = [
    # Money-center banks
    "JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "COF", "MTB",
    # Investment banks / brokers
    "GS", "MS", "SCHW",
    # Card networks (Sharia status debated; treated as non-halal here)
    "V", "MA", "AXP", "DFS",
    # Asset managers / alternative asset
    "BLK", "BX", "KKR", "APO", "TROW",
    # Insurance
    "BRK.B", "AIG", "MET", "PRU", "ALL", "TRV", "PGR", "CB", "AFL", "HIG",
    # Trust / custody banks
    "BK", "STT", "NTRS",
]

NON_HALAL_MIDCAP = [
    # Regional banks
    "KEY", "RF", "CFG", "CMA", "FITB", "ZION", "HBAN", "WAL", "OZK",
    "WBS", "VLY", "CADE", "FHN",
    # Specialty finance / brokers
    "SF", "RJF", "JEF", "LPLA", "AB", "AMG", "EVR", "PJT",
    # Insurance mid-caps
    "AIZ", "GL", "RGA", "AFG", "KMPR", "RNR", "ESGR", "CINF",
]


# ============================================================
# Build the master catalog. Halal lists are mostly mega-cap;
# MIDCAP_UNIVERSE is curated halal mid-caps. Non-halal lists
# above complete the matrix.
# ============================================================
def _classify_halal_megas() -> set[str]:
    """The HALAL_UNIVERSE list mixes mega- and mid-cap halal names. We
    split by membership in MIDCAP_UNIVERSE — anything in HALAL_UNIVERSE
    that is NOT in MIDCAP_UNIVERSE is treated as mega for the dashboard.
    """
    midset = set(MIDCAP_UNIVERSE)
    return {s for s in HALAL_UNIVERSE if s not in midset}


_HALAL_MEGAS = _classify_halal_megas()
_HALAL_MIDS = set(MIDCAP_UNIVERSE) & set(HALAL_UNIVERSE)
_HALAL_MIDS_ONLY = set(MIDCAP_UNIVERSE) - set(HALAL_UNIVERSE)
# MIDCAP_UNIVERSE is curated halal — treat any pure-mid-cap entries as halal
_ALL_HALAL_MIDS = set(MIDCAP_UNIVERSE)


def _build_meta() -> dict[str, SymbolMeta]:
    out: dict[str, SymbolMeta] = {}
    for s in _HALAL_MEGAS:
        out[s] = {"cap": "mega", "halal": True}
    for s in _ALL_HALAL_MIDS:
        out[s] = {"cap": "mid", "halal": True}
    for s in NON_HALAL_MEGACAP:
        out[s] = {"cap": "mega", "halal": False}
    for s in NON_HALAL_MIDCAP:
        out[s] = {"cap": "mid", "halal": False}
    return out


SYMBOL_META: dict[str, SymbolMeta] = _build_meta()


def all_symbols() -> list[str]:
    """Every symbol the screener can rank, deduped, sorted."""
    return sorted(SYMBOL_META.keys())


def symbols_for(
    *,
    halal: bool | None = None,
    cap: CapTier | None = None,
) -> list[str]:
    """Filter the master catalog. ``None`` for either field means no filter."""
    out: list[str] = []
    for sym, meta in SYMBOL_META.items():
        if halal is not None and meta["halal"] != halal:
            continue
        if cap is not None and meta["cap"] != cap:
            continue
        out.append(sym)
    return sorted(out)


def meta_for(symbol: str) -> SymbolMeta | None:
    return SYMBOL_META.get(symbol)


# ============================================================
# Sector / industry classification — loaded lazily from
# config/sector_map.json (built by scripts/build_sector_map.py
# using yfinance). Used by the ranker to apply correlation caps.
# ============================================================
import json as _json
from pathlib import Path as _Path

_SECTOR_MAP_PATH = _Path(__file__).resolve().parent / "sector_map.json"
_SECTOR_CACHE: dict[str, dict[str, str | None]] | None = None


def _load_sector_map() -> dict[str, dict[str, str | None]]:
    global _SECTOR_CACHE
    if _SECTOR_CACHE is not None:
        return _SECTOR_CACHE
    if _SECTOR_MAP_PATH.exists():
        try:
            _SECTOR_CACHE = _json.loads(_SECTOR_MAP_PATH.read_text())
        except (OSError, _json.JSONDecodeError):
            _SECTOR_CACHE = {}
    else:
        _SECTOR_CACHE = {}
    return _SECTOR_CACHE


def industry_for(symbol: str) -> str | None:
    """Fine-grained industry (e.g. "Semiconductors", "Banks - Diversified").

    This is what the ranker uses for correlation caps — sector is too
    coarse (lumps semis + software + hardware as "Technology").
    """
    rec = _load_sector_map().get(symbol)
    return rec.get("industry") if rec else None


def sector_for(symbol: str) -> str | None:
    """Coarse GICS-style sector (Technology, Energy, etc.)."""
    rec = _load_sector_map().get(symbol)
    return rec.get("sector") if rec else None
