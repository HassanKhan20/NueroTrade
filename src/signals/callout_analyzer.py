"""Pattern miner over the callouts feed — finds statistically significant
edges and weaknesses, refuses to recommend changes on small samples.

The user's intent: improve the screener by LEARNING from past callouts —
but only when the evidence is real, not when one bad week makes us throw
out a whole industry.

Architecture:
  1. Walk reports/callouts_feed.jsonl + bars cache to compute per-callout
     outcomes (TP1/TP2/STOP/OPEN).
  2. For each dimension (grade, cap_tier, halal, industry, RSI bucket),
     compute bucket win-rate and avg return.
  3. Run binomial test on win-rate vs the overall baseline, Welch's t-test
     on avg-return vs zero.
  4. Gate findings: require n >= MIN_N AND p < ALPHA before flagging.
  5. Output recommendations as human-readable text — never auto-apply.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.signals.callout_outcomes import compute_accuracy, resolve_outcome

# ---- Significance gates -------------------------------------------------- #
MIN_N = 30                  # don't flag patterns on smaller samples
ALPHA = 0.05                # 95% confidence threshold
ALPHA_STRONG = 0.01         # 99% confidence threshold (highest conviction)


@dataclass
class Finding:
    dimension: str           # "grade", "industry", "cap_tier", ...
    bucket: str              # the value of that dimension
    n: int                   # number of resolved callouts in this bucket
    win_rate: float          # (TP1+TP2) / resolved
    avg_return_pct: float
    baseline_win_rate: float
    baseline_avg_return: float
    p_value_win_rate: float
    p_value_return: float
    significance: str        # "strong" | "significant" | "underpowered"
    direction: str           # "outperform" | "underperform" | "neutral"
    recommendation: str      # human-readable suggested adjustment


@dataclass
class AnalyzerReport:
    generated_at: str
    n_total_callouts: int
    n_resolved: int
    overall_win_rate: float
    overall_avg_return: float
    findings: list[Finding] = field(default_factory=list)
    insufficient_data: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ============================================================
def _outcomes_for_analysis(feed_path: Path, bars_dir: Path,
                            window_days: int = 90,
                            horizon_days: int = 21) -> list[dict]:
    """Returns resolved-outcome dicts ready for slicing.

    Source of truth (preferred): TP1_HIT / TP2_HIT / STOP_HIT events
    written to the feed by the live watcher. These are exact prices at
    exact times — much fresher than the daily-bar replay.

    Fallback: ``compute_accuracy`` replays daily bars when no live event
    exists for a symbol. Useful for backtested data without live cycles.
    """
    if not feed_path.exists():
        return []

    events: list[dict] = []
    try:
        with feed_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    # Window cutoff
    cutoff = datetime.now(timezone.utc).timestamp() - window_days * 86400

    # Build map (symbol, entry_ts) -> {entry_event, terminal_event}
    open_callouts: dict[tuple[str, str], dict] = {}
    terminals: dict[tuple[str, str], dict] = {}
    for ev in events:
        sym = ev.get("symbol")
        ts_str = ev.get("ts", "")
        kind = ev.get("event")
        if not sym or not kind or not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if ts.timestamp() < cutoff:
            continue

        if kind in ("NEW", "UPGRADED"):
            key = (sym, ts_str)
            open_callouts[key] = ev
        elif kind in ("TP1_HIT", "TP2_HIT", "STOP_HIT"):
            entry_ts = ev.get("entry_ts")
            if not entry_ts:
                continue
            key = (sym, entry_ts)
            prior = terminals.get(key)
            # Keep the most decisive event: STOP_HIT or TP2_HIT trumps TP1_HIT
            if prior is None:
                terminals[key] = ev
            else:
                # If prior was TP1_HIT and now TP2_HIT/STOP_HIT, override
                if prior.get("event") == "TP1_HIT" and kind != "TP1_HIT":
                    terminals[key] = ev

    rows: list[dict] = []
    for key, entry in open_callouts.items():
        sym, ets = key
        terminal = terminals.get(key)
        if terminal is None:
            continue  # still open, skip for stats
        out_kind = terminal["event"]
        outcome = {"TP1_HIT": "HIT_TP1", "TP2_HIT": "HIT_TP2",
                   "STOP_HIT": "HIT_STOP"}[out_kind]
        ret = terminal.get("return_pct")
        rows.append({
            "symbol": sym,
            "grade": entry.get("grade", "F"),
            "cap_tier": entry.get("cap_tier"),
            "halal": entry.get("halal"),
            "outcome": outcome,
            "return_pct": float(ret) if ret is not None else 0.0,
            "ts": ets,
            "resolved_ts": terminal.get("ts"),
        })
    return rows


# ============================================================
def _bucket_stats(rows: list[dict], baseline_wr: float,
                   baseline_avg: float, dim: str,
                   key_fn) -> tuple[list[Finding], list[dict]]:
    """Group ``rows`` by ``key_fn``, run tests, return findings + buckets
    that lacked sufficient data."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = key_fn(r)
        if key is None:
            continue
        grouped[str(key)].append(r)

    findings: list[Finding] = []
    insufficient: list[dict] = []
    for bucket, group in grouped.items():
        n = len(group)
        wins = sum(1 for r in group if r["outcome"] in ("HIT_TP1", "HIT_TP2"))
        rets = [r["return_pct"] for r in group
                if r.get("return_pct") is not None]
        wr = wins / n if n else 0.0
        avg = sum(rets) / len(rets) if rets else 0.0

        if n < MIN_N:
            insufficient.append({
                "dimension": dim, "bucket": bucket, "n": n,
                "win_rate": wr, "avg_return_pct": avg,
                "need_more": MIN_N - n,
            })
            continue

        # Binomial test on win-rate vs baseline
        try:
            res = stats.binomtest(wins, n, p=baseline_wr, alternative="two-sided")
            p_wr = float(res.pvalue)
        except Exception:  # noqa: BLE001
            p_wr = 1.0
        # One-sample t-test on returns vs baseline avg
        if len(rets) >= 2:
            try:
                t = stats.ttest_1samp(rets, popmean=baseline_avg, nan_policy="omit")
                p_ret = float(t.pvalue) if np.isfinite(t.pvalue) else 1.0
            except Exception:  # noqa: BLE001
                p_ret = 1.0
        else:
            p_ret = 1.0

        # Significance: use the stronger of the two p-values
        p_best = min(p_wr, p_ret)
        if p_best < ALPHA_STRONG:
            sig = "strong"
        elif p_best < ALPHA:
            sig = "significant"
        else:
            sig = "underpowered"

        # Direction = signed difference vs baseline (use return as tiebreaker)
        if wr > baseline_wr and avg > baseline_avg:
            direction = "outperform"
        elif wr < baseline_wr and avg < baseline_avg:
            direction = "underperform"
        else:
            direction = "mixed"

        # Recommendation text
        if sig == "underpowered":
            recommendation = "no action — pattern not statistically significant"
        elif direction == "outperform":
            recommendation = (f"OVER-weight: this bucket beats the universe "
                              f"(WR {wr*100:.0f}% vs {baseline_wr*100:.0f}%, "
                              f"avg {avg:+.2f}% vs {baseline_avg:+.2f}%)")
        elif direction == "underperform":
            recommendation = (f"UNDER-weight or exclude: this bucket "
                              f"under-performs (WR {wr*100:.0f}% vs "
                              f"{baseline_wr*100:.0f}%, "
                              f"avg {avg:+.2f}% vs {baseline_avg:+.2f}%)")
        else:
            recommendation = "mixed signal — keep monitoring"

        findings.append(Finding(
            dimension=dim, bucket=bucket, n=n,
            win_rate=wr, avg_return_pct=avg,
            baseline_win_rate=baseline_wr,
            baseline_avg_return=baseline_avg,
            p_value_win_rate=p_wr, p_value_return=p_ret,
            significance=sig, direction=direction,
            recommendation=recommendation,
        ))
    return findings, insufficient


def _industry_key(r: dict) -> str | None:
    """Look up the industry for a callout's symbol. Returns ``None`` to
    skip the row if the sector map isn't loaded."""
    sym = r.get("symbol")
    if not sym:
        return None
    try:
        from config.universe_meta import industry_for
        ind = industry_for(sym)
    except ImportError:
        return None
    return ind or "_unknown"


# ============================================================
def analyze(feed_path: Path, bars_dir: Path,
             window_days: int = 90, horizon_days: int = 21,
             min_n: int = MIN_N) -> AnalyzerReport:
    """Top-level: load outcomes, slice by every dimension, return a
    report with statistically gated findings."""
    global MIN_N
    MIN_N = min_n  # allow override for tests

    rows = _outcomes_for_analysis(feed_path, bars_dir,
                                    window_days=window_days,
                                    horizon_days=horizon_days)

    n_resolved = len(rows)
    if n_resolved == 0:
        return AnalyzerReport(
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_total_callouts=0, n_resolved=0,
            overall_win_rate=0.0, overall_avg_return=0.0,
            notes=["no resolved callouts in window — let the system run "
                   "more cycles before drawing conclusions"],
        )

    wins = sum(1 for r in rows if r["outcome"] in ("HIT_TP1", "HIT_TP2"))
    baseline_wr = wins / n_resolved
    rets = [r["return_pct"] for r in rows if r.get("return_pct") is not None]
    baseline_avg = sum(rets) / len(rets) if rets else 0.0

    findings: list[Finding] = []
    insufficient: list[dict] = []

    dimensions = [
        ("grade",     lambda r: r.get("grade")),
        ("cap_tier",  lambda r: r.get("cap_tier")),
        ("halal",     lambda r: "halal" if r.get("halal") else "non-halal"),
        ("industry",  _industry_key),
    ]
    for dim, key_fn in dimensions:
        f, ins = _bucket_stats(rows, baseline_wr, baseline_avg, dim, key_fn)
        findings.extend(f)
        insufficient.extend(ins)

    # Sort findings: strong & significant first, then by absolute return delta
    findings.sort(key=lambda f: (
        {"strong": 0, "significant": 1, "underpowered": 2}.get(f.significance, 3),
        -abs(f.avg_return_pct - baseline_avg),
    ))

    notes: list[str] = []
    if n_resolved < 100:
        notes.append(f"Only {n_resolved} resolved callouts — treat findings "
                     f"as PRELIMINARY. Add more cycles before applying.")
    if not findings and not insufficient:
        notes.append("No buckets had enough data for analysis. Wait for "
                     "the watcher to accumulate more resolved callouts.")

    return AnalyzerReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_total_callouts=n_resolved,  # only resolved are usable
        n_resolved=n_resolved,
        overall_win_rate=baseline_wr,
        overall_avg_return=baseline_avg,
        findings=findings,
        insufficient_data=insufficient,
        notes=notes,
    )


# ============================================================
def report_to_dict(r: AnalyzerReport) -> dict:
    """JSON-safe serialisation."""
    return {
        "generated_at": r.generated_at,
        "n_total_callouts": r.n_total_callouts,
        "n_resolved": r.n_resolved,
        "overall_win_rate": round(r.overall_win_rate, 4),
        "overall_avg_return": round(r.overall_avg_return, 4),
        "findings": [
            {
                "dimension": f.dimension, "bucket": f.bucket, "n": f.n,
                "win_rate": round(f.win_rate, 4),
                "avg_return_pct": round(f.avg_return_pct, 4),
                "baseline_win_rate": round(f.baseline_win_rate, 4),
                "baseline_avg_return": round(f.baseline_avg_return, 4),
                "p_value_win_rate": round(f.p_value_win_rate, 4),
                "p_value_return": round(f.p_value_return, 4),
                "significance": f.significance,
                "direction": f.direction,
                "recommendation": f.recommendation,
            }
            for f in r.findings
        ],
        "insufficient_data": r.insufficient_data,
        "notes": r.notes,
    }
