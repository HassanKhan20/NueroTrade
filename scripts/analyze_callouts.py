"""One-shot pattern analyzer: walk callouts feed + bars, surface
statistically significant edges and weaknesses.

NEVER auto-applies adjustments — only reports findings.

Run: python -m scripts.analyze_callouts
"""
from __future__ import annotations

import json
from pathlib import Path

from config.settings import get_settings
from src.signals.callout_analyzer import analyze, report_to_dict
from src.utils.logging import configure_logging, get_logger

log = get_logger("analyze_callouts")


def main() -> int:
    s = get_settings()
    configure_logging(s.log_level, s.logs_dir / "analyzer.log")

    feed = s.reports_dir / "callouts_feed.jsonl"
    bars = s.data_dir / "parquet_daily"
    report = analyze(feed, bars, window_days=90, horizon_days=21)

    out_path = s.reports_dir / "analyzer_report.json"
    out_path.write_text(json.dumps(report_to_dict(report), indent=2,
                                    default=str))

    print()
    print(f"=== CALLOUT ANALYZER (window 90d, horizon 21d) ===")
    print(f"Resolved callouts analysed: {report.n_resolved}")
    print(f"Overall win-rate          : {report.overall_win_rate*100:.1f}%")
    print(f"Overall avg return        : {report.overall_avg_return:+.2f}%")
    print()
    if report.notes:
        print("Notes:")
        for n in report.notes:
            print(f"  - {n}")
        print()

    if report.findings:
        print(f"Findings ({len(report.findings)}):")
        for f in report.findings:
            mark = {"strong": "**", "significant": "* ",
                    "underpowered": "  "}.get(f.significance, "  ")
            print(f"  {mark}[{f.significance}] {f.dimension}={f.bucket}  "
                  f"n={f.n:<3}  WR {f.win_rate*100:>5.1f}%  "
                  f"avg {f.avg_return_pct:+6.2f}%  "
                  f"p={min(f.p_value_win_rate, f.p_value_return):.3f}")
            print(f"     -> {f.recommendation}")
        print()

    if report.insufficient_data:
        print(f"Insufficient data ({len(report.insufficient_data)} buckets):")
        for ins in report.insufficient_data[:20]:
            print(f"  {ins['dimension']}={ins['bucket']:<35}  "
                  f"n={ins['n']:<2}  need {ins['need_more']} more")
        if len(report.insufficient_data) > 20:
            print(f"  ... and {len(report.insufficient_data)-20} more")
        print()

    print(f"Full report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
