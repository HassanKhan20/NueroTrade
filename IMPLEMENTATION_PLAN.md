# NueroTrade Implementation Plan

A phased, verifiable build of a validated 3-component ensemble auto-trading bot:
**Stocks-in-Play 5-min ORB** + **Intraday Momentum (Gao et al.)** + **Volatility Regime Overlay**, with a López-de-Prado-grade anti-overfitting harness, deployed via Alpaca.

The existing Streamlit/LSTM demo in `app.py`, `model.py`, `utils.py` will be **archived** to `legacy/` (kept for reference, not deleted) and the repo restructured around the new bot.

---

## Definition of Done (the whole project)

A live paper-trading bot that:

1. Implements all three strategy components with objective rules.
2. Has passed Combinatorial Purged Cross-Validation with **PBO < 15%**, **Deflated Sharpe > 0.95**, **OOS Sharpe ≥ 70% of in-sample** on 2016–2023 train and 2024–2025 holdout.
3. Has run 30 trading days in shadow mode with simulated-vs-actual fill divergence < 1 bp on 95% of trades.
4. Has a real-time dashboard showing equity, open positions, PnL, and component-level attribution.
5. Has hard daily/weekly loss kill-switches that have been verified via fault injection.

---

## Repository target structure

```
NueroTrade/
├── README.md                       # rewritten to match reality
├── IMPLEMENTATION_PLAN.md          # this file
├── pyproject.toml
├── requirements.txt
├── .env.example                    # ALPACA_KEY, ALPACA_SECRET, POLYGON_KEY
├── .gitignore                      # must ignore .env, data/, logs/
├── config/
│   ├── settings.py                 # Pydantic settings, loads .env
│   └── strategy_params.yaml        # all numeric thresholds (frozen after CPCV)
├── data/                           # gitignored — local 1-min bars cache
│   ├── parquet/
│   └── manifests/
├── src/
│   ├── data/
│   │   ├── ingest_alpaca.py
│   │   ├── ingest_polygon.py       # optional, higher quality 1-min
│   │   └── universe.py             # liquid-universe filter
│   ├── costs/
│   │   ├── commissions.py          # $0.0035/share model
│   │   └── slippage.py             # bid-ask + impact model
│   ├── strategies/
│   │   ├── base.py                 # Strategy ABC
│   │   ├── orb_stocks_in_play.py   # Component A
│   │   ├── intraday_momentum.py    # Component B
│   │   └── vol_regime.py           # Component C overlay
│   ├── ensemble/
│   │   ├── allocator.py            # 60/30/10 risk-budget split
│   │   └── kill_switch.py          # daily/weekly loss limits
│   ├── backtest/
│   │   ├── engine.py               # vectorbt wrapper
│   │   ├── cpcv.py                 # combinatorial purged CV
│   │   ├── metrics.py              # PBO, DSR, bootstrap CIs
│   │   └── reports.py              # HTML report generator
│   ├── live/
│   │   ├── alpaca_client.py
│   │   ├── order_manager.py        # bracket orders
│   │   ├── shadow_mode.py
│   │   └── runner.py               # main event loop
│   ├── dashboard/
│   │   ├── app.py                  # FastAPI server
│   │   └── templates/index.html
│   └── utils/
│       ├── clock.py                # NY time, killzone helpers
│       └── logging.py              # structured logs
├── legacy/                         # archived old code
│   ├── app.py
│   ├── model.py
│   └── utils.py
├── tests/
│   ├── test_orb.py
│   ├── test_intraday_momentum.py
│   ├── test_vol_regime.py
│   ├── test_cpcv.py
│   ├── test_metrics.py             # PBO + DSR golden tests
│   ├── test_kill_switch.py
│   └── fixtures/
│       └── sample_1min_bars.parquet
├── scripts/
│   ├── download_history.py
│   ├── run_backtest.py
│   ├── run_cpcv.py
│   ├── freeze_params.py            # writes strategy_params.yaml from CPCV winner
│   ├── run_holdout.py              # gate: 2024-2025
│   ├── run_shadow.py
│   └── run_live.py
└── reports/                        # gitignored — generated artifacts
    ├── cpcv_results/
    └── holdout/
```

---

## Phase 0 — Repo bootstrap (Day 1, ~2 hrs)

**Goal:** Clean slate, locked deps, working test runner, archived old code.

### Tasks
1. Create `legacy/` directory; move `app.py`, `model.py`, `utils.py` into it.
2. Replace `requirements.txt` with locked versions:
   ```
   alpaca-py==0.30.*
   vectorbt==0.26.*
   numba==0.59.*
   pandas==2.2.*
   numpy==1.26.*
   pyarrow==15.*
   pydantic==2.*
   pydantic-settings==2.*
   python-dotenv==1.*
   pyyaml==6.*
   fastapi==0.110.*
   uvicorn==0.27.*
   pytest==8.*
   pytest-cov==4.*
   pytest-xdist==3.*
   hypothesis==6.*
   scipy==1.13.*
   statsmodels==0.14.*
   matplotlib==3.8.*
   plotly==5.20.*
   structlog==24.*
   tenacity==8.*
   pandas-market-calendars==4.*
   ```
3. Create `.env.example` (no real secrets) and `.gitignore` (must ignore `.env`, `data/`, `reports/`, `logs/`, `__pycache__`, `.pytest_cache`).
4. Create empty `src/` package tree with `__init__.py` files.
5. Create `pyproject.toml` with `pytest`, `ruff`, `mypy` configs.
6. CI sanity: `pytest -q` runs and finds 0 tests successfully.

### Acceptance
- `python -c "import vectorbt, alpaca, pydantic"` succeeds.
- `pytest -q` exits 0.
- `git status` clean except for new files; no `.env` tracked.

---

## Phase 1 — Data layer (Days 2–4)

**Goal:** Reliable, idempotent download of 1-min bars for SP500 + Nasdaq100 universe (2016-01-01 → today), stored locally as Parquet, plus a daily VIX/SPY/QQQ feed.

### Tasks
1. `src/data/ingest_alpaca.py`:
   - Function `fetch_minute_bars(symbol, start, end) -> DataFrame` using alpaca-py.
   - Rate-limited to 200 req/min, retries on 429 with exponential backoff (tenacity).
   - Writes to `data/parquet/{symbol}/{year}.parquet` partitioned.
   - Idempotent: re-running skips already-downloaded windows.
2. `src/data/universe.py`:
   - `liquid_universe(date)` returns names with avg dollar volume ≥ $20M over trailing 20 trading days, price ≥ $5, not a leveraged ETF (hardcoded blocklist of 3x ETFs).
3. `scripts/download_history.py`:
   - Downloads 2016-01-01 → present for SP500 + QQQ100 + VIX + SPY + QQQ.
   - Estimated ~2-4 hours; checkpoints progress.
4. Tests in `tests/test_data.py`:
   - Hypothesis-based test that downloaded bars are monotonic in timestamp, no duplicates, no NaN closes.
   - Test that `liquid_universe` excludes TQQQ, SQQQ, SPXL, etc.
   - Test that re-running ingest is a no-op.

### Acceptance
- `data/parquet/AAPL/2023.parquet` exists, ~99k rows.
- `liquid_universe('2023-06-01')` returns 200–500 names.
- All data tests pass.

### Risk / mitigation
- Alpaca free tier has IEX-only data (lower quality); this is **OK for development** but log a TODO to upgrade to Polygon ($29/mo or via Alpaca Algo Trader Plus) before live trading.

---

## Phase 2 — Cost models (Day 5)

**Goal:** Realistic commission + slippage models that the backtest engine plugs into.

### Tasks
1. `src/costs/commissions.py`:
   - `commission(shares, price) -> float` returning `max(0.0035 * shares, 0.35)` — Zarattini-Aziz convention; even though Alpaca is $0, model it as a slippage buffer.
2. `src/costs/slippage.py`:
   - `slippage(symbol, side, shares, ref_price, bar) -> float` returning a fill price.
   - Model: half-spread + 0.5 bp impact; spread from bar high-low / 2 if bid-ask not available.
3. Tests:
   - `commission(100, 50)` == $0.35.
   - Slippage on a buy is always ≥ ref_price; on a sell, ≤ ref_price.
   - Property test: cost is monotonic in shares.

### Acceptance
- Tests pass, cost values match Zarattini-Aziz paper conventions to within $0.001.

---

## Phase 3 — Strategy primitives (Days 6–10)

**Goal:** Three pure, deterministic strategy functions that take a DataFrame and emit signals + sizes. No I/O, no broker calls. Fully unit-tested.

### 3A. ORB Stocks-in-Play (`src/strategies/orb_stocks_in_play.py`)

Function signature:
```python
def orb_signals(
    bars_1min: dict[str, pd.DataFrame],   # symbol -> 1-min OHLCV for one day
    rel_vol_window: int = 14,
    top_n: int = 20,
    atr_window: int = 14,
    stop_atr_pct: float = 0.10,
    risk_per_trade: float = 0.01,
    max_leverage: float = 4.0,
    equity: float,
) -> list[Trade]
```
**Rules (frozen from Zarattini-Aziz):**
- 9:30–9:35 ET = opening range candle.
- Universe selection at 9:35: top-20 by `(today_premkt_to_935_volume) / (avg_premkt_to_935_volume_over_14_days)`.
- If OR candle green (close > open): place **buy-stop at OR-high**.
- If OR candle red: **sell-stop at OR-low**.
- Stop = entry ± `0.10 × ATR_14d` (in price terms).
- Size = `min(floor(0.01 × equity / stop_distance), shares_for_4x_leverage_cap)`.
- Time exit: 15:55 ET.
- No profit target.

**Tests** (`tests/test_orb.py`):
- Synthetic bars where AAPL's first candle is green and breaks high at 9:42 → emits long entry at OR-high price, stop at entry − 0.10×ATR, size matching 1% risk.
- If price never breaks OR-high, no trade.
- 4x leverage cap binds when many concurrent trades.
- Universe selection deterministic given same input.

### 3B. Intraday Momentum (`src/strategies/intraday_momentum.py`)

```python
def intraday_momentum_signals(
    bars_1min: pd.DataFrame,   # SPY or QQQ, full day
    prev_close: float,
    equity: float,
    risk_pct: float = 0.005,
) -> Trade | None
```
**Rules (Gao-Han-Li-Zhou, AND-filter variant):**
- At 14:55 ET, compute `r1 = (close_10:00 − prev_close) / prev_close`.
- Compute `r12 = (close_14:30 − close_14:00) / close_14:00`.
- Long at 15:00 if `r1 > 0 AND r12 > 0`; short if both negative; else flat.
- Exit at 15:55 ET.
- Size: `(0.005 × equity) / (ATR_5min × √11)`.

**Tests:**
- r1 > 0, r12 < 0 → no trade.
- Both > 0 → long signal at 15:00 close.
- Position size scales inversely with realized vol.

### 3C. Vol-Regime Overlay (`src/strategies/vol_regime.py`)

```python
def regime_multiplier(
    vix_today: float, vix_history: pd.Series,
    spy_returns_20d: pd.Series, spy_5y_history: pd.Series,
) -> float   # in {0.0, 0.5, 1.0}
```
**Rules:**
- `vix_pct = percentile_rank(vix_today, vix_history[-252:])`.
- `vol_z = zscore(realized_vol_20d, spy_5y_history)`.
- If `vix_pct > 0.97`: return 0.0 (no trading).
- Elif `vix_pct > 0.90 OR vol_z > 2`: return 0.5.
- Else: return 1.0.

**Tests:** golden cases for each branch.

### Acceptance for Phase 3
- Three pure-function modules, each with >90% line coverage.
- All tests pass; `pytest -q` < 30 s.

---

## Phase 4 — Backtest engine (Days 11–14)

**Goal:** A fast, vectorized backtester that consumes the strategy primitives and produces equity curves with realistic costs.

### Tasks
1. `src/backtest/engine.py`:
   - `class Backtester` wrapping vectorbt.
   - Inputs: date range, universe loader, list of strategy callables, cost models, initial equity, allocator (60/30/10).
   - Output: `BacktestResult` with equity curve, trade log, attribution by component, daily PnL.
2. `src/backtest/metrics.py`:
   - `sharpe(returns)`, `max_drawdown(equity)`, `profit_factor(trades)`, `win_rate(trades)`, `avg_r(trades)`.
   - `deflated_sharpe(sr, n_trials, n_obs, skew, kurt)` — Bailey & López de Prado 2014 formula.
   - `probabilistic_sharpe(sr, n_obs, skew, kurt, sr_benchmark=0)`.
   - `bootstrap_ci(returns, metric_fn, n_boot=5000)`.
3. `src/backtest/reports.py`:
   - HTML report: equity curve, drawdown chart, monthly heatmap, attribution by component, trade-distribution histogram.
4. `scripts/run_backtest.py`:
   - CLI: `python scripts/run_backtest.py --start 2016-01-01 --end 2023-12-31 --components A,B,C`.
   - Writes `reports/backtest_<timestamp>/`.
5. Tests:
   - Golden: replay 5 days of synthetic data, assert equity curve matches hand-computed values to 4 decimal places.
   - DSR golden against the López-de-Prado paper's worked example.

### Acceptance
- Full 2016-2023 backtest runs in < 15 minutes.
- Component-A-only Sharpe within ±0.3 of Zarattini-Aziz published 2.81 (allow drift due to Alpaca data quality).
- Reports render in browser.

---

## Phase 5 — Anti-overfitting harness (Days 15–18)

**Goal:** Combinatorial Purged Cross-Validation + PBO + DSR, gating any parameter freeze.

### Tasks
1. `src/backtest/cpcv.py`:
   - `cpcv_split(returns, n_splits=6, embargo_pct=0.01)` per López-de-Prado AFML Ch. 7.
   - `pbo(in_sample_perf, out_sample_perf)` — probability of overfitting.
2. `scripts/run_cpcv.py`:
   - Hyperparam grid (8 combos for ORB: stop_atr_pct ∈ {0.08, 0.10, 0.12, 0.15} × top_n ∈ {10, 20}).
   - For each split: train on in-sample, evaluate on out-sample, store metrics.
   - Compute PBO and DSR across the trial matrix.
3. `scripts/freeze_params.py`:
   - **Gate logic — fail-closed:**
     - PBO < 0.15
     - DSR > 0.95
     - Median OOS Sharpe ≥ 0.70 × median IS Sharpe
   - If any fails: exit non-zero, do **not** write params file.
   - If all pass: write `config/strategy_params.yaml` with the *median* of top-quartile param sets (NOT the best — best is overfit by definition).

### Acceptance
- CPCV runs and produces a results table.
- Test with deliberately overfit synthetic strategy: gates correctly reject (PBO > 0.5).
- Test with genuine edge (random walk + small drift): gates correctly accept.

---

## Phase 6 — Holdout test (Day 19) — HARD GATE

**Goal:** Run frozen parameters on **untouched** 2024-01-01 → today data. This dataset must never have been seen during Phases 4–5.

### Tasks
1. `scripts/run_holdout.py`:
   - Loads frozen `strategy_params.yaml`.
   - Backtests on 2024-2025 only.
   - Writes `reports/holdout/` with all metrics.
2. **Pass criteria:**
   - OOS Sharpe ≥ 0.70 × in-sample Sharpe.
   - Max drawdown ≤ 1.5× in-sample max drawdown.
   - Win rate within ±5pp of in-sample.
   - No single month worse than −12%.

### If holdout fails
- Do NOT tune to fix it. Document the failure, revisit Phase 5 hyperparam grid (it's a sign of overfitting), then re-freeze.
- Holdout can only be "used" once. Any second pass requires extending the holdout window forward in time.

### Acceptance
- Pass criteria met **OR** documented failure with revised plan.

---

## Phase 7 — Live infrastructure (Days 20–24)

**Goal:** Production-grade Alpaca client, order manager, kill switches, dashboard.

### Tasks
1. `src/live/alpaca_client.py`:
   - Thin async wrapper around alpaca-py with retries, structured logging.
   - Connection-loss handling: on disconnect, flatten everything, halt.
2. `src/live/order_manager.py`:
   - Bracket orders (entry + stop). 15:55 force-close handled by a separate timer task.
   - Idempotent order IDs (UUID per signal) to prevent duplicate fills on retry.
3. `src/ensemble/kill_switch.py`:
   - Daily loss > 2.5% → halt rest of day, flatten.
   - Weekly loss > 5% → halt until manually re-enabled (touch a `bot_state.json: "enabled": true` flag).
   - Per-symbol max position sanity cap.
4. `src/live/runner.py`:
   - Async event loop: market clock → universe screen → signal generation → sizing → ordering → state persistence.
   - Writes `bot_state.json` every 5 s.
5. `src/dashboard/app.py`:
   - FastAPI reads `bot_state.json`, serves equity, open positions, today's PnL, component attribution, kill-switch status.
   - HTMX-based auto-refresh every 2 s.
6. Tests with `pytest-asyncio`:
   - Order-manager submits exactly one bracket per signal under retries (idempotency).
   - Kill switch fires when synthetic equity drops 2.5%.
   - Disconnect mid-trade → flatten path is taken.

### Acceptance
- All tests pass.
- Dashboard renders against a fake `bot_state.json`.
- Kill switch verified by fault injection: manually set equity to trigger value → bot halts within 1 second.

---

## Phase 8 — Shadow mode (Days 25–34, 30 trading days minimum)

**Goal:** Run end-to-end against live market data without sending real orders. Compare to a parallel backtest of the same data.

### Tasks
1. `src/live/shadow_mode.py`:
   - Identical to live runner, but `submit_order` is a no-op that logs to `logs/shadow_orders.jsonl`.
2. Each evening, `scripts/compare_shadow.py`:
   - Reads shadow orders + that day's bars.
   - Reruns backtest engine on that single day with same data.
   - Asserts: every shadow order matches a backtest order; price within 1 bp on 95% of trades.
   - Writes a daily diff report.

### Acceptance
- 30 consecutive trading days completed.
- ≥ 95% trade match rate.
- No unhandled exceptions in `logs/runner.log`.

### If divergence > 5%
- Identify cause (data feed lag, clock drift, order-event handling). Fix. Restart 30-day clock.

---

## Phase 9 — Paper trading (Days 35–94, 60 days)

**Goal:** Real Alpaca paper account. 5% of intended capital at first; scale up only after weekly review confirms the system is behaving.

### Tasks
1. Switch `ALPACA_PAPER=true` and run `scripts/run_live.py`.
2. Weekly review checklist (`docs/weekly_review.md`):
   - Sharpe of the week vs. expectation band.
   - Drawdown vs. expectation band.
   - Any kill-switch trips?
   - Any unhandled exceptions?
3. Scaling rule:
   - Weeks 1-4: 5% of intended capital.
   - Weeks 5-8: 25% if no red flags.
   - Weeks 9-12: 100% only if Sharpe > 0.5 over the period AND max DD < 15%.

### Acceptance
- 60 days clean.
- Sharpe > 0.5 (lower bar than backtest; sample is small).
- No kill-switch trips beyond expected vol-regime ones.

---

## Phase 10 — Live capital (gate, manual decision)

**Not automated.** After paper passes, this is a **human decision**: review all artifacts (CPCV report, holdout report, shadow diff, paper review). If you proceed:

- Start with **5% of intended capital**, real money.
- Only after 60 more days clean, scale to full capital.
- Always keep the kill switches; never remove them "to give the strategy room."

---

## Cross-cutting concerns

### Secrets management
- All credentials in `.env`, loaded by `pydantic-settings`.
- `.env` in `.gitignore`. CI uses GitHub Actions secrets.

### Logging
- Structured JSON logs via `structlog` to `logs/runner.log`.
- One line per: signal, order submit, fill, kill-switch event, exception.

### Reproducibility
- Every backtest writes its config + git SHA + data manifest hash to its report directory.
- Anyone should be able to re-run any historical backtest exactly.

### Time correctness
- Use `pandas-market-calendars` for NYSE calendar + holidays.
- All internal timestamps stored as UTC; convert to America/New_York only at signal-time logic and display.
- Half-days handled (1:00 PM close): force-flat at 12:55, no afternoon momentum trade.

### Data quality
- Daily check: any symbol with > 5 minutes of missing bars during RTH gets dropped from that day's universe.
- VIX and SPY data must be present every trading day; else halt.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Alpaca IEX-only data quality during dev | Acceptable for backtest; upgrade to Polygon before paper trading |
| Strategy decay (2-5 yr half-life per literature) | Quarterly re-validation via Phase 5 harness on rolling window; auto-disable if Sharpe drops |
| Personal psychology (quitting on normal DD) | Anchor expectations to realistic column in Section 5; pre-commit to weekly review only |
| Single point of failure (laptop down) | Run on a small VPS once paper-validated; systemd unit auto-restarts |
| Bug in cost model = inflated backtest | Phase 8 shadow mode is the sanity check; divergence > 1 bp is a stop-the-line event |
| Holdout used too early | Holdout dir is read-only outside Phase 6; CI check enforces |

---

## Timeline summary

| Phase | Days | Cumulative |
|---|---|---|
| 0. Bootstrap | 1 | 1 |
| 1. Data layer | 3 | 4 |
| 2. Cost models | 1 | 5 |
| 3. Strategy primitives | 5 | 10 |
| 4. Backtest engine | 4 | 14 |
| 5. Anti-overfitting harness | 4 | 18 |
| 6. Holdout (HARD GATE) | 1 | 19 |
| 7. Live infrastructure | 5 | 24 |
| 8. Shadow mode | 30 (calendar) | 54 |
| 9. Paper trading | 60 (calendar) | 114 |
| 10. Live capital decision | manual | — |

**Realistic build time to paper-ready: ~24 working days. Realistic time to live capital: ~4 months.**

---

## Immediate next step

Phase 0 (today): archive the legacy demo, lock dependencies, bootstrap the new package layout. After that, Phase 1 starts with `src/data/ingest_alpaca.py`.
