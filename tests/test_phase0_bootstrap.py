"""Phase 0 sentinel: project skeleton is wired and importable."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_layout_exists() -> None:
    expected_dirs = [
        "src/data",
        "src/costs",
        "src/strategies",
        "src/ensemble",
        "src/backtest",
        "src/live",
        "src/dashboard",
        "src/utils",
        "config",
        "scripts",
        "legacy",
        "tests",
    ]
    missing = [d for d in expected_dirs if not (REPO_ROOT / d).is_dir()]
    assert not missing, f"missing directories: {missing}"


def test_legacy_archive_present() -> None:
    for f in ("app.py", "model.py", "utils.py"):
        assert (REPO_ROOT / "legacy" / f).is_file(), f"legacy/{f} missing"


def test_no_top_level_legacy_modules() -> None:
    for f in ("app.py", "model.py", "utils.py"):
        assert not (REPO_ROOT / f).exists(), (
            f"{f} should be archived under legacy/, not at repo root"
        )


def test_settings_loads_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)
    from config.settings import Settings

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.initial_equity == 25_000.0
    assert s.risk_pct_per_trade == 0.01
    assert s.max_leverage == 4.0
    assert s.daily_loss_limit == 0.025
    assert s.weekly_loss_limit == 0.05
    assert s.alpaca_paper is True


def test_strategy_params_yaml_present_and_unfrozen() -> None:
    import yaml

    params = yaml.safe_load((REPO_ROOT / "config" / "strategy_params.yaml").read_text())
    assert params["frozen"] is False, (
        "strategy_params.yaml must start unfrozen; only Phase 5 may freeze it"
    )
    assert "orb_stocks_in_play" in params
    assert "intraday_momentum" in params
    assert "vol_regime" in params
    assert "allocator" in params

    a = params["allocator"]
    total = a["weight_orb"] + a["weight_intraday_momentum"] + a["cash_reserve"]
    assert abs(total - 1.0) < 1e-9, f"allocator weights must sum to 1, got {total}"


def test_gitignore_blocks_secrets_and_data() -> None:
    text = (REPO_ROOT / ".gitignore").read_text()
    for entry in (".env", "data/parquet/", "reports/", "logs/", "bot_state.json"):
        assert entry in text, f".gitignore missing entry: {entry}"


def test_env_example_does_not_contain_real_keys() -> None:
    text = (REPO_ROOT / ".env.example").read_text()
    assert "your_alpaca_key_here" in text
    assert "PK" not in text.split("ALPACA_KEY=")[1].split("\n")[0]
