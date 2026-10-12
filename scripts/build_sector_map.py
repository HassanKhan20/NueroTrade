"""One-shot: build config/sector_map.json mapping symbol -> {sector, industry}.

We use ``industry`` rather than ``sector`` for the correlation cap because
"Technology" lumps semis, software, and hardware together — but only semis
move as one trade. Industries like "Semiconductors", "Software—Application",
"Banks—Diversified" are tight enough to be a real correlation bucket.

Run: python -m scripts.build_sector_map
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import yfinance as yf

from config.universe_meta import all_symbols

OUT_PATH = Path(__file__).resolve().parents[1] / "config" / "sector_map.json"


def main() -> int:
    symbols = all_symbols()
    print(f"Fetching sector/industry for {len(symbols)} symbols...")
    out: dict[str, dict[str, str | None]] = {}
    failed: list[str] = []
    for i, sym in enumerate(symbols, 1):
        try:
            info = yf.Ticker(sym).info
            out[sym] = {
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
            print(f"  [{i:>3}/{len(symbols)}] {sym:<6} "
                  f"sector={out[sym]['sector']!r:30}  "
                  f"industry={out[sym]['industry']!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:>3}/{len(symbols)}] {sym}: FAILED ({e})")
            failed.append(sym)
            out[sym] = {"sector": None, "industry": None}
        time.sleep(0.05)
    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True))
    print()
    print(f"Wrote {OUT_PATH}")
    print(f"Resolved: {len(symbols) - len(failed)} / {len(symbols)}")
    if failed:
        print(f"Failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
