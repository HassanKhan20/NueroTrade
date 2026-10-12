"""One-shot: build config/earnings_cache.json from yfinance.

For each symbol in the master universe, fetch the last ~8 quarters of
earnings announcements with consensus + actual + surprise%. The PEAD
signal reads this cache to compute SUE + drift weighting.

Run: python -m scripts.build_earnings_cache
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import yfinance as yf

from config.universe_meta import all_symbols

OUT_PATH = Path(__file__).resolve().parents[1] / "config" / "earnings_cache.json"


def main() -> int:
    symbols = all_symbols()
    print(f"Fetching earnings history for {len(symbols)} symbols...")
    out: dict[str, list[dict]] = {}
    ok = fail = 0
    for i, sym in enumerate(symbols, 1):
        try:
            ed = yf.Ticker(sym).earnings_dates
            if ed is None or ed.empty:
                out[sym] = []
                fail += 1
                continue
            events = []
            for ts, row in ed.iterrows():
                actual = row.get("Reported EPS")
                consensus = row.get("EPS Estimate")
                surprise = row.get("Surprise(%)")
                # Skip future / unreported rows (no Reported EPS yet)
                if actual is None or (isinstance(actual, float) and math.isnan(actual)):
                    continue
                if consensus is None or (isinstance(consensus, float) and math.isnan(consensus)):
                    continue
                # Normalise surprise: yfinance reports as % already
                events.append({
                    "announcement_date": ts.strftime("%Y-%m-%d"),
                    "actual_eps": float(actual),
                    "consensus_eps": float(consensus),
                    "surprise_pct": float(surprise) if surprise is not None and not (
                        isinstance(surprise, float) and math.isnan(surprise)
                    ) else (float(actual) - float(consensus)) / max(abs(float(consensus)), 1e-9) * 100,
                })
            out[sym] = events
            ok += 1
            print(f"  [{i:>3}/{len(symbols)}] {sym:<6}  {len(events)} events")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:>3}/{len(symbols)}] {sym}: FAILED ({type(e).__name__})")
            out[sym] = []
            fail += 1
        time.sleep(0.05)

    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    print()
    print(f"Wrote {OUT_PATH}")
    print(f"Symbols with events: {ok}, no data / failed: {fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
