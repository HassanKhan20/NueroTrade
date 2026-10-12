"""One-shot self-grader: pull current Alpaca quotes, compare against each
callout's entry/stop/TP1/TP2, print a verdict.

Run: python -m scripts.grade_self
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from config.settings import get_settings


def main() -> int:
    events = []
    with open("reports/callouts_feed.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    by_sym = {}
    for e in events:
        by_sym[e["symbol"]] = e
    events = list(by_sym.values())

    print(f"callouts to grade: {len(events)}")
    if not events:
        print("no callouts to grade.")
        return 0
    ts_first = events[0]["ts"]
    emit_dt = datetime.fromisoformat(ts_first.replace("Z", "+00:00"))
    hours = (datetime.now(timezone.utc) - emit_dt).total_seconds() / 3600
    print(f"emitted: {ts_first}  ({hours:.1f}h ago)")

    s = get_settings()
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest

    client = StockHistoricalDataClient(s.alpaca_key, s.alpaca_secret)
    syms = [e["symbol"] for e in events]
    quotes = client.get_stock_latest_trade(
        StockLatestTradeRequest(symbol_or_symbols=syms),
    )

    results = []
    for e in events:
        sym = e["symbol"]
        q = quotes.get(sym)
        if q is None:
            continue
        last = float(q.price)
        entry = float(e["entry_price"])
        stop = float(e["stop_price"])
        tp1 = float(e["take_profit_1"])
        tp2 = float(e["take_profit_2"])
        ret_pct = (last / entry - 1.0) * 100
        if last >= tp2:
            outcome = "HIT_TP2"
        elif last >= tp1:
            outcome = "HIT_TP1"
        elif last <= stop:
            outcome = "HIT_STOP"
        else:
            outcome = "OPEN"
        results.append({
            "symbol": sym, "grade": e["grade"],
            "cap_tier": e.get("cap_tier"), "halal": e.get("halal"),
            "entry": entry, "last": last, "stop": stop, "tp1": tp1,
            "ret_pct": ret_pct, "outcome": outcome,
        })

    total = len(results)
    out_counts = Counter(r["outcome"] for r in results)
    stops = out_counts.get("HIT_STOP", 0)
    opens = out_counts.get("OPEN", 0)
    tp1s = out_counts.get("HIT_TP1", 0)
    tp2s = out_counts.get("HIT_TP2", 0)
    returns = [r["ret_pct"] for r in results]
    above_water = sum(1 for r in returns if r > 0)
    avg_ret = sum(returns) / len(returns) if returns else 0

    print()
    print(f"=== SELF-GRADE (interim, {hours:.1f}h since callouts) ===")
    print(f"hit limits        : HIT_TP1={tp1s}  HIT_TP2={tp2s}  HIT_STOP={stops}")
    print(f"still inside band : OPEN={opens}")
    print(f"above entry now   : {above_water}/{total} "
          f"({100*above_water/total:.1f}%)")
    print(f"avg interim return: {avg_ret:+.2f}%")
    best = max(results, key=lambda r: r["ret_pct"])
    worst = min(results, key=lambda r: r["ret_pct"])
    print(f"best : {best['symbol']} {best['ret_pct']:+.2f}%")
    print(f"worst: {worst['symbol']} {worst['ret_pct']:+.2f}%")

    print()
    print("By grade:")
    by_grade = defaultdict(list)
    for r in results:
        by_grade[r["grade"]].append(r)
    for g in ("A+", "A", "B", "C", "D", "F"):
        rs = by_grade.get(g, [])
        if not rs:
            continue
        avg = sum(r["ret_pct"] for r in rs) / len(rs)
        aw = sum(1 for r in rs if r["ret_pct"] > 0)
        print(f"  {g:>2}: n={len(rs):2}  avg {avg:+.2f}%  "
              f"above-water {aw}/{len(rs)}")

    print()
    print("By cap tier:")
    by_cap = defaultdict(list)
    for r in results:
        by_cap[r["cap_tier"]].append(r)
    for c, rs in by_cap.items():
        avg = sum(r["ret_pct"] for r in rs) / len(rs)
        aw = sum(1 for r in rs if r["ret_pct"] > 0)
        print(f"  {c:>4}: n={len(rs):2}  avg {avg:+.2f}%  "
              f"above-water {aw}/{len(rs)}")

    print()
    print("By halal:")
    by_h = defaultdict(list)
    for r in results:
        by_h["halal" if r["halal"] else "non-halal"].append(r)
    for h, rs in by_h.items():
        avg = sum(r["ret_pct"] for r in rs) / len(rs)
        aw = sum(1 for r in rs if r["ret_pct"] > 0)
        print(f"  {h:>10}: n={len(rs):2}  avg {avg:+.2f}%  "
              f"above-water {aw}/{len(rs)}")

    print()
    print("Top 10 leaders:")
    for r in sorted(results, key=lambda r: -r["ret_pct"])[:10]:
        print(f"  {r['symbol']:<5} [{r['grade']:>2}] "
              f"${r['entry']:>7.2f} -> ${r['last']:>7.2f}  "
              f"{r['ret_pct']:+5.2f}%  {r['outcome']}")
    print()
    print("Bottom 10 laggards:")
    for r in sorted(results, key=lambda r: r["ret_pct"])[:10]:
        print(f"  {r['symbol']:<5} [{r['grade']:>2}] "
              f"${r['entry']:>7.2f} -> ${r['last']:>7.2f}  "
              f"{r['ret_pct']:+5.2f}%  {r['outcome']}")

    snap = {
        "emitted_at": ts_first,
        "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hours_elapsed": round(hours, 1),
        "total": total, "hit_tp1": tp1s, "hit_tp2": tp2s,
        "hit_stop": stops, "open": opens,
        "above_water": above_water,
        "avg_return_pct": round(avg_ret, 3),
        "by_grade": {
            g: {"n": len(rs),
                "avg": round(sum(r["ret_pct"] for r in rs) / len(rs), 2)}
            for g, rs in by_grade.items()
        },
        "by_cap": {
            c: {"n": len(rs),
                "avg": round(sum(r["ret_pct"] for r in rs) / len(rs), 2)}
            for c, rs in by_cap.items()
        },
        "by_halal": {
            h: {"n": len(rs),
                "avg": round(sum(r["ret_pct"] for r in rs) / len(rs), 2)}
            for h, rs in by_h.items()
        },
        "detail": sorted(results, key=lambda r: -r["ret_pct"]),
    }
    with open("reports/self_grade_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, default=str)
    print()
    print("Snapshot written: reports/self_grade_snapshot.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
