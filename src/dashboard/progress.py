"""Aggregates real-time progress signals from the filesystem:

* download_status  — counts parquet files vs. expected; total size; per-symbol
                     year grid; reads the last few logs/ingest.log events.
* backtest_status  — lists every reports/backtests/*/summary.json with metrics.
* cpcv_status      — reports/cpcv/cpcv_summary.json if present.
* gate_status      — config/strategy_params.yaml frozen + git_sha.
* bot_status       — bot_state.json if running.

All read-only. Safe to call from any HTTP handler.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "parquet"
LOGS_DIR = REPO_ROOT / "logs"
REPORTS_DIR = REPO_ROOT / "reports"
PARAMS_PATH = REPO_ROOT / "config" / "strategy_params.yaml"
BOT_STATE_PATH = REPO_ROOT / "bot_state.json"
BACKTEST_PROGRESS_PATH = REPORTS_DIR / "backtest_progress.json"
SCREENER_LATEST_PATH = REPORTS_DIR / "screener_latest.json"
CALLOUTS_FEED_PATH = REPORTS_DIR / "callouts_feed.jsonl"
WILLIAMS_LATEST_PATH = REPORTS_DIR / "williams_latest.json"
WILLIAMS_FEED_PATH = REPORTS_DIR / "williams_feed.jsonl"


@dataclass
class SymbolProgress:
    symbol: str
    years_done: list[int]
    rows_total: int
    bytes_total: int


@dataclass
class DownloadStatus:
    started_at: str | None
    last_event_at: str | None
    n_files: int
    n_expected: int | None
    pct_done: float
    total_bytes: int
    total_rows: int
    symbols: list[SymbolProgress]
    recent_events: list[dict[str, Any]]
    failed_events: list[dict[str, Any]]


@dataclass
class BacktestSummary:
    label: str
    path: str
    n_days: int
    n_trades: int
    terminal_equity: float
    total_return: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    component_attribution: dict[str, float]
    written_at: str


@dataclass
class CPCVStatus:
    present: bool
    pbo: float | None = None
    deflated_sharpe: float | None = None
    best_strategy: str | None = None
    passed: bool | None = None
    written_at: str | None = None


@dataclass
class GateStatus:
    params_frozen: bool
    git_sha: str | None
    frozen_at: str | None


@dataclass
class BacktestProgress:
    """Live progress of a currently-running backtest (None if idle)."""

    label: str
    started_at: str
    updated_at: str
    processed: int
    total: int
    pct_done: float
    current_date: str | None
    equity: float
    n_trades: int
    attribution: dict[str, float]
    regime: str


@dataclass
class LiveAccount:
    """Real Alpaca paper account snapshot, fetched on every dashboard poll
    so the user sees actual fills/positions, not just backtest history."""

    equity: float
    cash: float
    buying_power: float
    todays_pnl: float
    todays_pnl_pct: float
    n_positions: int
    positions: list[dict[str, Any]]
    paper: bool


@dataclass
class ScreenerPicks:
    """Latest output of scripts/screener.py — surfaced on the dashboard."""
    as_of: str
    universe: str
    n_universe: int
    n_qualified: int
    picks: list[dict[str, Any]]
    generated_at: str


@dataclass
class CalloutsFeed:
    """Tail of reports/callouts_feed.jsonl — newest event last in the file."""
    n_events: int
    last_ts: str | None
    events: list[dict[str, Any]]  # newest first, capped at N


@dataclass
class AccuracyBucketDict:
    label: str
    n: int
    n_resolved: int
    n_open: int
    n_tp1: int
    n_tp2: int
    n_stop: int
    n_expired: int
    avg_return_pct: float
    hit_rate: float


@dataclass
class CalloutAccuracySnapshot:
    """Self-graded accuracy of past NEW/UPGRADED callouts."""
    generated_at: str
    window_days: int
    n_total: int
    overall: AccuracyBucketDict
    by_grade: list[AccuracyBucketDict]
    by_cap: list[AccuracyBucketDict]
    by_halal: list[AccuracyBucketDict]
    recent: list[dict[str, Any]]


@dataclass
class ProgressSnapshot:
    ts: str
    download: DownloadStatus
    backtests: list[BacktestSummary]
    cpcv: CPCVStatus
    gate: GateStatus
    bot_state: dict[str, Any] | None
    running_backtest: BacktestProgress | None
    live_account: LiveAccount | None = None
    screener: ScreenerPicks | None = None
    callouts_feed: CalloutsFeed | None = None
    callout_accuracy: CalloutAccuracySnapshot | None = None
    learning_report: dict[str, Any] | None = None
    williams: dict[str, Any] | None = None
    williams_feed: CalloutsFeed | None = None


# --------------------------------------------------------------------------- #
def _read_jsonl_tail(path: Path, n: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in text.splitlines()[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _scan_parquet() -> tuple[list[SymbolProgress], int, int]:
    if not DATA_DIR.exists():
        return [], 0, 0
    syms: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    total_files = 0
    for sym_dir in sorted(DATA_DIR.iterdir()):
        if not sym_dir.is_dir():
            continue
        years: list[int] = []
        rows = 0
        size = 0
        for f in sorted(sym_dir.glob("*.parquet")):
            year_match = re.match(r"(\d{4})\.parquet$", f.name)
            if year_match:
                years.append(int(year_match.group(1)))
            try:
                size += f.stat().st_size
                # Cheap row-count: parquet file size / ~100 bytes per row is good enough
                # for the dashboard. We avoid actually opening every file.
                rows += max(1, f.stat().st_size // 100)
                total_files += 1
            except OSError:
                pass
        syms[sym_dir.name] = {
            "symbol": sym_dir.name,
            "years_done": sorted(set(years)),
            "rows_total": rows,
            "bytes_total": size,
        }
        total_bytes += size
    return [SymbolProgress(**s) for s in syms.values()], total_bytes, total_files


def _detect_expected_files(events: list[dict[str, Any]]) -> int | None:
    """Best-effort: scan recent events for any 'ingest' write/skip lines and
    derive the unique (symbol, year) tuples we've ever seen — that gives us
    a rolling expected count without needing the user to declare it."""
    seen = set()
    for e in events:
        s = e.get("symbol")
        y = e.get("year")
        if s and y:
            seen.add((s, int(y)))
    return len(seen) or None


def collect_download_status(expected: int | None = None) -> DownloadStatus:
    log_path = LOGS_DIR / "ingest.log"
    events = _read_jsonl_tail(log_path, n=2000)
    syms, total_bytes, n_files = _scan_parquet()
    total_rows = sum(s.rows_total for s in syms)

    failed = [e for e in events if e.get("msg") == "ingest.failed" or e.get("level") == "ERROR"]
    started_at = events[0]["ts"] if events else None
    last_event_at = events[-1]["ts"] if events else None

    pct = 0.0
    if expected:
        pct = min(100.0, n_files / expected * 100.0)

    return DownloadStatus(
        started_at=started_at,
        last_event_at=last_event_at,
        n_files=n_files,
        n_expected=expected,
        pct_done=pct,
        total_bytes=total_bytes,
        total_rows=total_rows,
        symbols=syms,
        recent_events=events[-15:],
        failed_events=failed[-10:],
    )


def collect_backtests() -> list[BacktestSummary]:
    bt_dir = REPORTS_DIR / "backtests"
    if not bt_dir.exists():
        return []
    out: list[BacktestSummary] = []
    for d in sorted(bt_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        sp = d / "summary.json"
        if not sp.exists():
            continue
        try:
            data = json.loads(sp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            BacktestSummary(
                label=str(data.get("label", d.name)),
                path=str(d),
                n_days=int(data.get("n_days", 0)),
                n_trades=int(data.get("n_trades", 0)),
                terminal_equity=float(data.get("terminal_equity", 0.0)),
                total_return=float(data.get("total_return", 0.0)),
                sharpe=float(data.get("sharpe", 0.0)),
                max_drawdown=float(data.get("max_drawdown", 0.0)),
                profit_factor=float(data.get("profit_factor", 0.0) or 0.0),
                win_rate=float(data.get("win_rate", 0.0) or 0.0),
                component_attribution=dict(data.get("component_attribution", {})),
                written_at=str(data.get("written_at", "")),
            )
        )
    return out


def collect_cpcv() -> CPCVStatus:
    p = REPORTS_DIR / "cpcv" / "cpcv_summary.json"
    if not p.exists():
        return CPCVStatus(present=False)
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return CPCVStatus(present=False)
    return CPCVStatus(
        present=True,
        pbo=d.get("pbo"),
        deflated_sharpe=d.get("deflated_sharpe"),
        best_strategy=d.get("best_strategy"),
        passed=d.get("passed"),
        written_at=d.get("ts"),
    )


def collect_gate() -> GateStatus:
    if not PARAMS_PATH.exists():
        return GateStatus(params_frozen=False, git_sha=None, frozen_at=None)
    try:
        p = yaml.safe_load(PARAMS_PATH.read_text())
    except (OSError, yaml.YAMLError):
        return GateStatus(params_frozen=False, git_sha=None, frozen_at=None)
    return GateStatus(
        params_frozen=bool(p.get("frozen", False)),
        git_sha=p.get("git_sha"),
        frozen_at=p.get("frozen_at"),
    )


def collect_bot_state() -> dict[str, Any] | None:
    if not BOT_STATE_PATH.exists():
        return None
    try:
        return json.loads(BOT_STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def collect_running_backtest() -> BacktestProgress | None:
    if not BACKTEST_PROGRESS_PATH.exists():
        return None
    try:
        d = json.loads(BACKTEST_PROGRESS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return BacktestProgress(
        label=str(d.get("label", "")),
        started_at=str(d.get("started_at", "")),
        updated_at=str(d.get("updated_at", "")),
        processed=int(d.get("processed", 0)),
        total=int(d.get("total", 0)),
        pct_done=float(d.get("pct_done", 0.0)),
        current_date=d.get("current_date"),
        equity=float(d.get("equity", 0.0)),
        n_trades=int(d.get("n_trades", 0)),
        attribution=dict(d.get("attribution", {})),
        regime=str(d.get("regime", "")),
    )


def collect_live_account() -> LiveAccount | None:
    """Fetch the actual Alpaca paper account state. Cached 5 seconds to
    avoid hammering the API while still feeling real-time."""
    global _LIVE_CACHE, _LIVE_CACHE_AT
    now = datetime.now(timezone.utc).timestamp()
    if _LIVE_CACHE is not None and (now - _LIVE_CACHE_AT) < 5.0:
        return _LIVE_CACHE
    try:
        from config.settings import get_settings
        from src.live.alpaca_client import AlpacaBrokerClient

        s = get_settings()
        if not s.alpaca_key or not s.alpaca_secret:
            return None
        broker = AlpacaBrokerClient(s.alpaca_key, s.alpaca_secret, paper=s.alpaca_paper)
        # account() doesn't include yesterday's equity — fetch via raw client
        broker._ensure()  # type: ignore[attr-defined]
        a = broker._client.get_account()  # type: ignore[union-attr]
        equity = float(a.equity)
        last_eq = float(a.last_equity) if a.last_equity else equity
        todays_pnl = equity - last_eq
        todays_pnl_pct = (todays_pnl / last_eq) if last_eq > 0 else 0.0
        positions_raw = broker.positions()
        positions = []
        for p in positions_raw:
            try:
                px = broker._client.get_open_position(p.symbol)  # type: ignore[union-attr]
                positions.append({
                    "symbol": p.symbol,
                    "qty": int(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(px.current_price) if px.current_price else 0.0,
                    "market_value": float(px.market_value) if px.market_value else 0.0,
                    "unrealized_pl": float(px.unrealized_pl) if px.unrealized_pl else 0.0,
                    "unrealized_plpc": float(px.unrealized_plpc) if px.unrealized_plpc else 0.0,
                })
            except Exception:  # noqa: BLE001
                positions.append({
                    "symbol": p.symbol, "qty": int(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": 0.0, "market_value": 0.0,
                    "unrealized_pl": 0.0, "unrealized_plpc": 0.0,
                })
        out = LiveAccount(
            equity=equity, cash=float(a.cash),
            buying_power=float(a.buying_power),
            todays_pnl=todays_pnl, todays_pnl_pct=todays_pnl_pct,
            n_positions=len(positions), positions=positions,
            paper=bool(s.alpaca_paper),
        )
        _LIVE_CACHE = out
        _LIVE_CACHE_AT = now
        return out
    except Exception:  # noqa: BLE001
        return None


_LIVE_CACHE: LiveAccount | None = None
_LIVE_CACHE_AT: float = 0.0


def collect_callouts_feed(max_events: int = 50) -> CalloutsFeed | None:
    """Tail the JSONL feed, return the newest ``max_events`` events."""
    if not CALLOUTS_FEED_PATH.exists():
        return None
    try:
        with CALLOUTS_FEED_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None
    if not lines:
        return CalloutsFeed(n_events=0, last_ts=None, events=[])
    parsed: list[dict[str, Any]] = []
    for line in lines[-max_events * 2:]:  # extra slack for malformed lines
        line = line.strip()
        if not line:
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    parsed.reverse()  # newest first
    last_ts = parsed[0].get("ts") if parsed else None
    return CalloutsFeed(
        n_events=len(lines),
        last_ts=last_ts,
        events=parsed[:max_events],
    )


def _tail_feed(path: Path, max_events: int = 50) -> CalloutsFeed | None:
    """Generic JSONL feed tailer (newest first)."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None
    if not lines:
        return CalloutsFeed(n_events=0, last_ts=None, events=[])
    parsed: list[dict[str, Any]] = []
    for line in lines[-max_events * 2:]:
        line = line.strip()
        if not line:
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    parsed.reverse()
    return CalloutsFeed(
        n_events=len(lines),
        last_ts=parsed[0].get("ts") if parsed else None,
        events=parsed[:max_events],
    )


def collect_williams() -> dict[str, Any] | None:
    """Latest Williams breakout snapshot (williams_latest.json)."""
    if not WILLIAMS_LATEST_PATH.exists():
        return None
    try:
        return json.loads(WILLIAMS_LATEST_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def collect_williams_feed() -> CalloutsFeed | None:
    # Larger cap so the live callout feed can be scrolled back through
    # a full session of events, not just the last handful.
    return _tail_feed(WILLIAMS_FEED_PATH, max_events=200)


_ACCURACY_CACHE: CalloutAccuracySnapshot | None = None
_ACCURACY_AT: float = 0.0


def collect_callout_accuracy(
    window_days: int = 90, horizon_days: int = 21,
) -> CalloutAccuracySnapshot | None:
    """Compute self-graded accuracy of past callouts.

    Cached for 60s — recomputing on every dashboard tick is wasteful and
    the underlying data only changes when the watcher writes new feed
    lines (every 15 min) or new daily bars land.
    """
    global _ACCURACY_CACHE, _ACCURACY_AT
    now_ts = datetime.now(timezone.utc).timestamp()
    if _ACCURACY_CACHE is not None and (now_ts - _ACCURACY_AT) < 60.0:
        return _ACCURACY_CACHE

    try:
        from src.signals.callout_outcomes import compute_accuracy
        bars_dir = REPO_ROOT / "data" / "parquet_daily"
        acc = compute_accuracy(
            CALLOUTS_FEED_PATH, bars_dir,
            window_days=window_days, horizon_days=horizon_days,
        )
    except Exception:  # noqa: BLE001
        return None

    def _b(b: Any) -> AccuracyBucketDict:
        return AccuracyBucketDict(
            label=b.label, n=b.n, n_resolved=b.n_resolved, n_open=b.n_open,
            n_tp1=b.n_tp1, n_tp2=b.n_tp2, n_stop=b.n_stop,
            n_expired=b.n_expired,
            avg_return_pct=b.avg_return_pct, hit_rate=b.hit_rate,
        )

    snap = CalloutAccuracySnapshot(
        generated_at=acc.generated_at,
        window_days=acc.window_days,
        n_total=acc.n_total,
        overall=_b(acc.overall),
        by_grade=[_b(x) for x in acc.by_grade],
        by_cap=[_b(x) for x in acc.by_cap],
        by_halal=[_b(x) for x in acc.by_halal],
        recent=acc.recent,
    )
    _ACCURACY_CACHE = snap
    _ACCURACY_AT = now_ts
    return snap


_LEARNING_CACHE: dict[str, Any] | None = None
_LEARNING_AT: float = 0.0


def collect_learning_report() -> dict[str, Any] | None:
    """Pattern-analyzer report: statistically gated edges/weaknesses in
    past callouts. Cached 5 min — the underlying data only changes when
    new feed events land."""
    global _LEARNING_CACHE, _LEARNING_AT
    now_ts = datetime.now(timezone.utc).timestamp()
    if _LEARNING_CACHE is not None and (now_ts - _LEARNING_AT) < 300.0:
        return _LEARNING_CACHE
    try:
        from src.signals.callout_analyzer import analyze, report_to_dict
        bars = REPO_ROOT / "data" / "parquet_daily"
        rpt = analyze(CALLOUTS_FEED_PATH, bars,
                       window_days=90, horizon_days=21)
        _LEARNING_CACHE = report_to_dict(rpt)
        _LEARNING_AT = now_ts
        return _LEARNING_CACHE
    except Exception:  # noqa: BLE001
        return None


def collect_screener() -> ScreenerPicks | None:
    if not SCREENER_LATEST_PATH.exists():
        return None
    try:
        d = json.loads(SCREENER_LATEST_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return ScreenerPicks(
        as_of=str(d.get("as_of", "")),
        universe=str(d.get("universe", "")),
        n_universe=int(d.get("n_universe", 0)),
        n_qualified=int(d.get("n_qualified", 0)),
        picks=list(d.get("picks", [])),
        generated_at=str(d.get("generated_at", "")),
    )


def snapshot(expected: int | None = None) -> ProgressSnapshot:
    return ProgressSnapshot(
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        download=collect_download_status(expected),
        backtests=collect_backtests(),
        cpcv=collect_cpcv(),
        gate=collect_gate(),
        bot_state=collect_bot_state(),
        running_backtest=collect_running_backtest(),
        live_account=collect_live_account(),
        screener=collect_screener(),
        callouts_feed=collect_callouts_feed(),
        callout_accuracy=collect_callout_accuracy(),
        learning_report=collect_learning_report(),
        williams=collect_williams(),
        williams_feed=collect_williams_feed(),
    )


def snapshot_dict(expected: int | None = None) -> dict[str, Any]:
    """Serialise the snapshot to a JSON-safe dict.

    NaN and Infinity values are converted to None — JSON has no native
    representation for them, and FastAPI's default encoder rejects them.
    """
    import math as _m

    s = snapshot(expected)

    def _clean(v: Any) -> Any:
        if isinstance(v, float) and (_m.isnan(v) or _m.isinf(v)):
            return None
        return v

    def to_d(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_d(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [to_d(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_d(v) for k, v in obj.items()}
        return _clean(obj)

    return to_d(s)
