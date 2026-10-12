from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.dashboard import app as dashboard_app
from src.live.state import BotState, write_state


def test_health_endpoint() -> None:
    client = TestClient(dashboard_app.app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_state_endpoint_404_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard_app, "STATE_PATH", tmp_path / "no_state.json")
    client = TestClient(dashboard_app.app)
    r = client.get("/api/state")
    assert r.status_code == 404


def test_state_endpoint_returns_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "bot_state.json"
    write_state(BotState(equity=42.0, today_trades=2, kill_switch_state="ok"), p)
    monkeypatch.setattr(dashboard_app, "STATE_PATH", p)
    client = TestClient(dashboard_app.app)
    r = client.get("/api/state")
    assert r.status_code == 200
    body = r.json()
    assert body["equity"] == 42.0
    assert body["today_trades"] == 2


def test_classic_renders_no_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard_app, "STATE_PATH", tmp_path / "no_state.json")
    client = TestClient(dashboard_app.app)
    r = client.get("/classic")
    assert r.status_code == 200
    assert "Waiting for bot_state.json" in r.text


def test_classic_renders_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "bot_state.json"
    write_state(BotState(equity=99_000.0, today_trades=5), p)
    monkeypatch.setattr(dashboard_app, "STATE_PATH", p)
    client = TestClient(dashboard_app.app)
    r = client.get("/classic")
    assert r.status_code == 200
    assert "$99,000.00" in r.text


def test_index_serves_progress_dashboard() -> None:
    client = TestClient(dashboard_app.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "NEUROTRADE" in r.text.upper()
