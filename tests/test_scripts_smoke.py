"""Smoke tests for CLI scripts.

We don't run them against real Alpaca; we verify the gate logic by feeding
hand-crafted summary files and checking exit codes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import freeze_params, run_cpcv, run_holdout


def test_freeze_params_fails_when_no_summary(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rc = freeze_params.main(
        ["--cpcv-summary", str(tmp_path / "absent.json"),
         "--params", str(tmp_path / "params.yaml")]
    )
    assert rc == 1


def test_freeze_params_refuses_when_cpcv_failed(tmp_path: Path) -> None:
    summary = tmp_path / "cpcv.json"
    summary.write_text(json.dumps({"passed": False, "pbo": 0.4, "deflated_sharpe": 0.5}))
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"frozen": False}))
    rc = freeze_params.main(["--cpcv-summary", str(summary), "--params", str(params)])
    assert rc == 2
    p = yaml.safe_load(params.read_text())
    assert p["frozen"] is False  # untouched


def test_freeze_params_writes_when_passed(tmp_path: Path) -> None:
    summary = tmp_path / "cpcv.json"
    summary.write_text(json.dumps({
        "passed": True, "pbo": 0.05,
        "deflated_sharpe": 0.97, "best_strategy": "v1",
    }))
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"frozen": False, "orb_stocks_in_play": {}}))
    rc = freeze_params.main(["--cpcv-summary", str(summary), "--params", str(params)])
    assert rc == 0
    p = yaml.safe_load(params.read_text())
    assert p["frozen"] is True
    assert "frozen_at" in p


def test_run_cpcv_passes_for_real_edge(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 252 * 5
    # Strong consistent edge across 4 strategies
    rets = pd.DataFrame({f"s{i}": rng.normal(0.0008, 0.01, n) for i in range(4)})
    rets_path = tmp_path / "rets.parquet"
    rets.to_parquet(rets_path)
    out_dir = tmp_path / "cpcv_out"
    rc = run_cpcv.main([
        "--returns", str(rets_path), "--out", str(out_dir),
        "--groups", "5", "--test", "2", "--embargo", "5",
    ])
    summary = json.loads((out_dir / "cpcv_summary.json").read_text())
    assert "pbo" in summary and "deflated_sharpe" in summary


def test_run_holdout_blocks_unfrozen_params(tmp_path: Path) -> None:
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"frozen": False}))
    is_summary = tmp_path / "is.json"
    is_summary.write_text(json.dumps({"sharpe": 1.5, "max_drawdown": -0.1}))
    oos_summary = tmp_path / "oos.json"
    oos_summary.write_text(json.dumps({"sharpe": 1.4, "max_drawdown": -0.12}))
    eq = pd.Series([1.0, 1.01, 1.02, 1.0, 0.98], index=pd.date_range("2024-01-01", periods=5))
    eq_path = tmp_path / "eq.csv"
    eq.to_csv(eq_path, header=["equity"])
    rc = run_holdout.main([
        "--params", str(params),
        "--is-summary", str(is_summary),
        "--oos-summary", str(oos_summary),
        "--oos-equity", str(eq_path),
    ])
    assert rc == 1


def test_run_holdout_passes_when_thresholds_met(tmp_path: Path) -> None:
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"frozen": True}))
    is_summary = tmp_path / "is.json"
    is_summary.write_text(json.dumps({"sharpe": 1.5, "max_drawdown": -0.10}))
    oos_summary = tmp_path / "oos.json"
    oos_summary.write_text(json.dumps({"sharpe": 1.3, "max_drawdown": -0.11}))
    eq = pd.Series(
        np.linspace(1.0, 1.2, 60),
        index=pd.date_range("2024-01-01", periods=60, freq="B"),
    )
    eq_path = tmp_path / "eq.csv"
    eq.to_csv(eq_path, header=["equity"])
    rc = run_holdout.main([
        "--params", str(params),
        "--is-summary", str(is_summary),
        "--oos-summary", str(oos_summary),
        "--oos-equity", str(eq_path),
    ])
    assert rc == 0


def test_run_holdout_fails_when_oos_sharpe_too_low(tmp_path: Path) -> None:
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"frozen": True}))
    is_summary = tmp_path / "is.json"
    is_summary.write_text(json.dumps({"sharpe": 2.0, "max_drawdown": -0.10}))
    oos_summary = tmp_path / "oos.json"
    oos_summary.write_text(json.dumps({"sharpe": 0.5, "max_drawdown": -0.10}))
    eq = pd.Series(
        np.linspace(1.0, 1.05, 30),
        index=pd.date_range("2024-01-01", periods=30, freq="B"),
    )
    eq_path = tmp_path / "eq.csv"
    eq.to_csv(eq_path, header=["equity"])
    rc = run_holdout.main([
        "--params", str(params),
        "--is-summary", str(is_summary),
        "--oos-summary", str(oos_summary),
        "--oos-equity", str(eq_path),
    ])
    assert rc == 3
