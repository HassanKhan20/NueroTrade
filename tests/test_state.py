from __future__ import annotations

from pathlib import Path

from src.live.state import BotState, read_state, write_state


def test_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "bot_state.json"
    s = BotState(equity=42_000.0, today_trades=3, kill_switch_state="ok")
    write_state(s, p)
    s2 = read_state(p)
    assert s2 is not None
    assert s2.equity == 42_000.0
    assert s2.today_trades == 3
    assert s2.updated_at  # populated by stamp()


def test_read_missing_returns_none(tmp_path: Path) -> None:
    assert read_state(tmp_path / "absent.json") is None
