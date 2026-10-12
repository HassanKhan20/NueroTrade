# NueroTrade

A validated 3-component ensemble auto-trading bot for US equities, deployed via Alpaca.

**Mandate: long-only equities. No shorts. No options. No leveraged ETFs.**

The long-only restriction is enforced by an invariant in [src/strategies/base.py](src/strategies/base.py): constructing a `Side.SHORT` Trade raises `ValueError`.

**Components**
1. **Stocks-in-Play 5-min ORB (long-only)** — Zarattini, Barbon & Aziz (SSRN 2024); only green opening-range candles trigger entries.
2. **Intraday Momentum (long-only)** — Gao, Han, Li & Zhou (JFE 2018); only positive-on-both r1 / r12 days enter.
3. **Volatility Regime Overlay** — VIX-percentile / realized-vol risk sizing.

**Validation harness:** Combinatorial Purged Cross-Validation, Probability of Backtest Overfitting (gate: PBO < 15%), Deflated Sharpe Ratio (gate: DSR > 0.95), and an untouchable 2024–2025 holdout (gate: OOS Sharpe ≥ 70% of in-sample).

> The full design rationale lives in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

---

## Status

**All build-time phases complete; 138 tests passing.** All bot machinery is in place:

- Phase 0 — repo bootstrap, deps locked, legacy archived.
- Phase 1 — Alpaca minute-bar ingest (`src/data/ingest_alpaca.py`), liquid-universe filter, idempotent caching.
- Phase 2 — commission ($0.0035/share, $0.35 floor) and slippage (half-spread + impact) cost models.
- Phase 3 — three long-only strategy primitives + invariant-enforcing `Trade` dataclass.
- Phase 4 — event-driven backtest engine (`src/backtest/engine.py`) producing equity curves, trade logs, component attribution.
- Phase 5 — CPCV harness (`src/backtest/cpcv.py`) + PBO/DSR gates (`scripts/run_cpcv.py`, `scripts/freeze_params.py`).
- Phase 6 — holdout gate script (`scripts/run_holdout.py`) with Sharpe-retention, max-DD multiple, and worst-month thresholds.
- Phase 7 — live infrastructure: Alpaca client wrapper, idempotent order manager (deterministic client_order_id), session runner with kill switches, FastAPI dashboard, shadow-mode broker.
- Phase 8 scaffolding — `scripts/run_shadow.py`, `scripts/compare_shadow.py` (≥95% trade-match gate), and synthetic-data fixtures so end-to-end tests run without real bars.
- Phase 9 scaffolding — `scripts/run_live.py` with hard fail on unfrozen params; `docs/ops_runbook.md` and `docs/weekly_review.md` operational checklists.

Phases 8-10 (30-day shadow, 60-day paper, live capital) are operational milestones — they require running the bot against real data, not writing more code.

The original demo is preserved in [legacy/](legacy/) for reference; it is not part of the new bot.

---

## Project layout

```
NueroTrade/
├── README.md                       — this file
├── IMPLEMENTATION_PLAN.md          — phased build plan, acceptance criteria
├── pyproject.toml                  — package + tool config
├── requirements.txt                — locked deps
├── .env.example                    — copy to .env and fill in
├── .gitignore                      — secrets/data/reports excluded
├── config/
│   ├── settings.py                 — pydantic settings, loads .env
│   └── strategy_params.yaml        — frozen params (written by Phase 5)
├── src/
│   ├── data/        — Alpaca/Polygon ingest, liquid-universe screen
│   ├── costs/       — commission + slippage models
│   ├── strategies/  — ORB, intraday momentum, vol regime
│   ├── ensemble/    — capital allocator, kill switches
│   ├── backtest/    — event-driven engine, CPCV, metrics, reports
│   ├── live/        — Alpaca client, order manager, runner
│   ├── dashboard/   — FastAPI status UI
│   └── utils/       — clock, logging
├── legacy/          — archived Streamlit/LSTM demo
├── tests/           — unit + integration tests
└── scripts/         — download_history, run_backtest, run_cpcv, freeze_params,
                       run_holdout, run_shadow, run_live
```

---

## Local setup

> **Python 3.10 required** (pinned for pyarrow / pandas wheel compatibility). Use `pyenv`, `conda`, or download a 3.10 build from python.org.

```bash
python3.10 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then fill in ALPACA_KEY / ALPACA_SECRET
pytest -q
```

Phase 0 sentinel tests verify the skeleton; later phases add their own.

---

## Build phases (see IMPLEMENTATION_PLAN.md for detail)

| Phase | What | Gate |
|---|---|---|
| 0 | Bootstrap | sentinel tests pass ✅ |
| 1 | Data ingest | 2016–today minute bars cached |
| 2 | Cost models | commission/slippage tests pass |
| 3 | Strategy primitives | unit tests, >90% coverage |
| 4 | Backtest engine | full backtest in <15 min |
| 5 | Anti-overfitting harness | PBO < 0.15, DSR > 0.95 |
| 6 | Holdout | OOS Sharpe ≥ 0.70 × IS |
| 7 | Live infra | kill switch + dashboard verified |
| 8 | Shadow mode (30 days) | trade-match rate ≥ 95% |
| 9 | Paper (60 days) | clean run, scaling rule |
| 10 | Live capital | manual decision |

---

## Safety

- All API keys live in `.env` (gitignored). Never commit secrets.
- Hard daily loss limit (default 2.5%) and weekly loss limit (default 5%) halt the bot automatically.
- Holdout (2024–2025) is read-only outside Phase 6 — never tune against it.

---

## Disclaimer

This is research software. Past backtest performance does not predict future returns. Trade at your own risk.
