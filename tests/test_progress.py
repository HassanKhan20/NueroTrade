"""Tests for the progress dashboard endpoints + scanner."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from src.dashboard import app as dashboard_app
from src.dashboard import progress as progress_mod


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the progress module at a temp directory."""
    monkeypatch.setattr(progress_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(progress_mod, "DATA_DIR", tmp_path / "data" / "parquet")
    monkeypatch.setattr(progress_mod, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(progress_mod, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(progress_mod, "PARAMS_PATH", tmp_path / "config" / "strategy_params.yaml")
    monkeypatch.setattr(progress_mod, "BOT_STATE_PATH", tmp_path / "bot_state.json")
    monkeypatch.setattr(
        progress_mod, "BACKTEST_PROGRESS_PATH", tmp_path / "reports" / "backtest_progress.json"
    )
    return tmp_path


def test_snapshot_with_empty_repo(isolated_repo: Path) -> None:
    snap = progress_mod.snapshot_dict()
    assert snap["download"]["n_files"] == 0
    assert snap["download"]["symbols"] == []
    assert snap["backtests"] == []
    assert snap["cpcv"]["present"] is False
    assert snap["gate"]["params_frozen"] is False
    assert snap["bot_state"] is None
    assert snap["running_backtest"] is None


def test_snapshot_reads_running_backtest(isolated_repo: Path) -> None:
    rdir = isolated_repo / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "backtest_progress.json").write_text(json.dumps({
        "label": "ict_run",
        "started_at": "2026-05-04T08:00:00+00:00",
        "updated_at": "2026-05-04T08:01:30+00:00",
        "processed": 87,
        "total": 249,
        "pct_done": 34.94,
        "current_date": "2025-09-12",
        "equity": 24830.55,
        "n_trades": 14,
        "attribution": {"orb": 0.0, "intraday_momentum": 0.0, "ict": -169.45},
        "regime": "ict",
    }))
    snap = progress_mod.snapshot_dict()
    rb = snap["running_backtest"]
    assert rb is not None
    assert rb["label"] == "ict_run"
    assert rb["pct_done"] == pytest.approx(34.94)
    assert rb["processed"] == 87
    assert rb["total"] == 249
    assert rb["regime"] == "ict"
    assert rb["attribution"]["ict"] == pytest.approx(-169.45)


def test_snapshot_running_backtest_corrupt_file_returns_none(
    isolated_repo: Path,
) -> None:
    rdir = isolated_repo / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "backtest_progress.json").write_text("not valid json{{{")
    snap = progress_mod.snapshot_dict()
    assert snap["running_backtest"] is None


def test_snapshot_counts_parquet_files(isolated_repo: Path) -> None:
    pdir = isolated_repo / "data" / "parquet" / "AAPL"
    pdir.mkdir(parents=True)
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [100.0]},
        index=pd.DatetimeIndex(["2025-01-02 14:30:00"], tz="UTC"),
    )
    df.to_parquet(pdir / "2025.parquet")
    df.to_parquet(pdir / "2026.parquet")

    snap = progress_mod.snapshot_dict(expected=4)
    assert snap["download"]["n_files"] == 2
    assert snap["download"]["pct_done"] == 50.0
    syms = snap["download"]["symbols"]
    assert len(syms) == 1
    assert syms[0]["symbol"] == "AAPL"
    assert syms[0]["years_done"] == [2025, 2026]
    assert snap["download"]["total_bytes"] > 0


def test_snapshot_reads_backtest_summary(isolated_repo: Path) -> None:
    bt = isolated_repo / "reports" / "backtests" / "demo_run"
    bt.mkdir(parents=True)
    (bt / "summary.json").write_text(json.dumps({
        "label": "demo",
        "n_days": 252,
        "n_trades": 48,
        "terminal_equity": 113000.0,
        "total_return": 0.13,
        "sharpe": 1.42,
        "max_drawdown": -0.08,
        "profit_factor": 1.8,
        "win_rate": 0.41,
        "component_attribution": {"orb": 9000.0, "intraday_momentum": 4000.0},
        "written_at": "2026-05-04T05:30:00+00:00",
    }))
    snap = progress_mod.snapshot_dict()
    assert len(snap["backtests"]) == 1
    bt_summary = snap["backtests"][0]
    assert bt_summary["sharpe"] == pytest.approx(1.42)
    assert bt_summary["component_attribution"]["orb"] == 9000.0


def test_snapshot_reads_cpcv(isolated_repo: Path) -> None:
    cp = isolated_repo / "reports" / "cpcv"
    cp.mkdir(parents=True)
    (cp / "cpcv_summary.json").write_text(json.dumps({
        "ts": "2026-05-04T05:00:00+00:00",
        "passed": True,
        "pbo": 0.08,
        "deflated_sharpe": 0.97,
        "best_strategy": "stop10_top20",
    }))
    snap = progress_mod.snapshot_dict()
    assert snap["cpcv"]["present"] is True
    assert snap["cpcv"]["passed"] is True
    assert snap["cpcv"]["pbo"] == pytest.approx(0.08)


def test_snapshot_reads_frozen_params(isolated_repo: Path) -> None:
    cdir = isolated_repo / "config"
    cdir.mkdir(parents=True)
    (cdir / "strategy_params.yaml").write_text(yaml.safe_dump({
        "frozen": True,
        "git_sha": "abcdef1234567890",
        "frozen_at": "2026-05-04T05:00:00+00:00",
    }))
    snap = progress_mod.snapshot_dict()
    assert snap["gate"]["params_frozen"] is True
    assert snap["gate"]["git_sha"] == "abcdef1234567890"


# ----- HTTP endpoint tests -----
def test_health_endpoint() -> None:
    client = TestClient(dashboard_app.app)
    assert client.get("/api/health").json() == {"ok": True}


def test_progress_endpoint_returns_json(isolated_repo: Path) -> None:
    client = TestClient(dashboard_app.app)
    r = client.get("/api/progress")
    assert r.status_code == 200
    body = r.json()
    assert "download" in body
    assert "backtests" in body
    assert "cpcv" in body
    assert "gate" in body


def test_progress_endpoint_with_expected_param(isolated_repo: Path) -> None:
    client = TestClient(dashboard_app.app)
    r = client.get("/api/progress?expected=66")
    assert r.status_code == 200
    assert r.json()["download"]["n_expected"] == 66


def test_index_serves_dashboard_html() -> None:
    client = TestClient(dashboard_app.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "NEUROTRADE" in r.text.upper()
    # Looks like the new dashboard, not the old minimal view
    assert "/api/progress" in r.text


def test_classic_endpoint_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.live.state import BotState, write_state
    p = tmp_path / "bot_state.json"
    write_state(BotState(equity=42_000.0), p)
    monkeypatch.setattr(dashboard_app, "STATE_PATH", p)
    client = TestClient(dashboard_app.app)
    r = client.get("/classic")
    assert r.status_code == 200
    assert "$42,000.00" in r.text
