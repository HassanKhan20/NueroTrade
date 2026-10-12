# Weekly review checklist

Run every Friday after close, before the weekend.

## 1. Performance
- [ ] Pull this week's equity curve from `bot_state.json` history (or reports/).
- [ ] Compute weekly Sharpe, weekly return, weekly max drawdown.
- [ ] Compare to the realistic expectation band: Sharpe 1.0-1.5 annualized → ~0.14-0.21 weekly equivalent.
- [ ] Anything outside ±2σ of expectation: flag for investigation, do NOT scale up.

## 2. Risk hygiene
- [ ] Did the daily kill switch trip on any day? If yes, was the loss within model expectations?
- [ ] Did the vol-regime overlay halve or halt size on any day? Was VIX percentile actually elevated?
- [ ] Any unhandled exceptions in `logs/runner.log` or `logs/live.log`?
- [ ] Any orders rejected by Alpaca? Why?

## 3. Component attribution
- [ ] ORB net P&L this week.
- [ ] Intraday momentum net P&L this week.
- [ ] Are both contributing? If one is consistently negative for 4+ weeks, flag for component-level CPCV refresh.

## 4. Data quality
- [ ] Spot-check any day with > 5 missing minutes for our names; halt if data feed degraded.
- [ ] VIX and SPY data present for every trading day.

## 5. Drift signals
- [ ] Number of OBR signals per day vs. backtest expectation.
- [ ] Average rel-volume of selected names vs. backtest expectation.
- [ ] Any obvious regime shift (rates, earnings season, macro event) that should pause for a re-validation cycle.

## 6. Decision
- [ ] Continue at current capital fraction.
- [ ] Scale up (only if Section 1 + Section 2 both clean for 4 weeks).
- [ ] Scale down or halt (manual `enabled: false` in `bot_state.json`).

## Sign-off
Operator: ________  Date: ________  Notes:
