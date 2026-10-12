from __future__ import annotations

from pathlib import Path

from src.live.shadow_mode import ShadowBroker


def test_shadow_records_orders(tmp_path: Path) -> None:
    log_path = tmp_path / "shadow.jsonl"
    sb = ShadowBroker(log_path=log_path, starting_equity=50_000.0)
    sb.submit_bracket_long("AAPL", 100, 200.0, 199.0, "coid-1")
    sb.submit_market_long("QQQ", 50, "coid-2")
    sb.flatten_all()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    assert "bracket_long" in lines[0]
    assert "market_long" in lines[1]
    assert "flatten_all" in lines[2]


def test_shadow_account_returns_starting_equity(tmp_path: Path) -> None:
    sb = ShadowBroker(log_path=tmp_path / "shadow.jsonl", starting_equity=12_345.0)
    a = sb.account()
    assert a.equity == 12_345.0
    assert a.cash == 12_345.0
