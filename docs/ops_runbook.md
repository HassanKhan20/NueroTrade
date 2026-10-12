# NueroTrade Ops Runbook

The order of operations from "I have data" to "I am live with capital."

## TL;DR — push-button path

Once `.env` is filled and Python 3.10 venv is active:

```
python -m scripts.check_connection                # verify keys + data feed
python -m scripts.quickstart --symbols AAPL MSFT NVDA AMZN META GOOGL ^
   TSLA AVGO LLY V JPM XOM UNH WMT MA PG JNJ HD COST ABBV CVX MRK NFLX ^
   KO PEP ADBE CRM ORCL AMD QCOM
```

`quickstart` runs: download → IS backtest → grid → CPCV gate → freeze → OOS → holdout gate. Exits non-zero at any failed gate. Re-runs are idempotent.

The detailed sections below explain what each step does.

## 0. Prerequisites
- Python 3.10 venv with `pip install -r requirements.txt`.
- `.env` filled in (copy from `.env.example`); never commit it.
- `pytest -q` is green.
- An Alpaca paper account (paid Alpaca Algo Trader Plus or Polygon $29/mo recommended for true minute data; Alpaca free tier is IEX-only and OK for development).

## 1. Bulk download history
```
python -m scripts.download_history --start 2016 --end 2025 \
    --symbols AAPL MSFT NVDA AMZN META GOOGL ... SPY QQQ
```
~2-4 hours on the free tier. Idempotent — re-run to top up new years.

## 2. In-sample backtest (2016-2023)
```
python -m scripts.run_backtest --start 2016-01-01 --end 2023-12-31 \
    --symbols <30 names> --label is_2016_2023
```
Inspect `reports/backtests/is_2016_2023_*/summary.json`. Sanity-check Sharpe is in the range you expect (1.0-2.0 net of slippage; less is honest, more is suspicious).

## 3. CPCV gate
```
python -m scripts.run_cpcv --returns reports/grid_returns.parquet
```
You build `grid_returns.parquet` by running step 2 with several parameter variants and stacking their daily returns into one wide DataFrame.

**Hard fail unless PBO < 0.15 AND DSR > 0.95.**

## 4. Freeze parameters
```
python -m scripts.freeze_params
```
Writes `frozen: true` and the git SHA into `config/strategy_params.yaml`. From this point on, edits to that file should land in PRs reviewed against the next CPCV run.

## 5. Holdout
The 2024-2025 window must NEVER have been used in 1-3.
```
python -m scripts.run_backtest --start 2024-01-01 --end 2025-12-31 \
    --symbols <same names> --label oos_holdout
python -m scripts.run_holdout \
    --is-summary  reports/backtests/is_2016_2023_<...>/summary.json \
    --oos-summary reports/backtests/oos_holdout_<...>/summary.json \
    --oos-equity  reports/backtests/oos_holdout_<...>/equity_curve.csv
```
**Hard fail** unless OOS Sharpe ≥ 0.7×IS, OOS max-DD ≤ 1.5×IS, worst OOS month ≥ −12%.

If holdout fails: do **not** tune to fix it. The window is burnt; extend the holdout forward in time before any second pass.

## 6. Shadow mode (30 trading days)
Cron at 09:34 NY each trading day:
```
python -m scripts.run_shadow --date $(date +%F) \
    --symbols <30 names>
```
Each evening:
```
python -m scripts.compare_shadow --date $(date +%F) --symbols <30 names>
```
Pass criterion: 30 consecutive days, ≥95% trade-match rate.

## 7. Paper trading (60 trading days, scaling)
```
python -m scripts.run_live --symbols <30 names>     # ALPACA_PAPER=true
```
- Weeks 1-4: 5% of intended capital.
- Weeks 5-8: 25% if no red flags.
- Weeks 9-12: 100% only if Sharpe > 0.5 over the period AND max DD < 15%.

## 8. Live capital
Manual decision only. Re-read `IMPLEMENTATION_PLAN.md` Phase 10 before flipping `ALPACA_PAPER=false`.

## Operational discipline
- Do NOT silence kill switches "to give the strategy room." If they trip, find out why before re-enabling.
- Daily loss > 2.5% halts intraday; weekly loss > 5% halts until manually re-enabled by editing `bot_state.json`.
- Re-run the CPCV harness quarterly on a rolling window. If DSR drops below 0.95, pause and re-validate.
