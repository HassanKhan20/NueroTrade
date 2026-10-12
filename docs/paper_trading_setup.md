# Paper Trading — Daily Scheduler Setup

The CSM strategy is now paper-trading on your Alpaca paper account. To keep
it running, you need a daily cron-style trigger. The runner is **idempotent**
— it only rebalances if 15 trading days have passed since the last one. Safe
to fire every weekday.

## What the runner does each invocation

1. Reads `logs/csm_state.json` (or creates it).
2. If < 15 trading days since last rebalance: **logs and exits.**
3. Otherwise:
   - Loads daily bars from `data/parquet_daily/` cache + tail-fetch any new days.
   - Pulls Alpaca paper account + current positions.
   - Computes 12-1 momentum ranking on the SP100 universe.
   - Filters by absolute-momentum (must have positive 12-1 return).
   - Picks top-5; computes inverse-vol weights.
   - Diffs against current positions.
   - Submits market orders to flatten exiting names + open new picks.
   - Persists state.

## Windows Task Scheduler — recommended setup

1. Open **Task Scheduler** (`taskschd.msc`).
2. **Create Basic Task** → name: "NeuroTrade CSM Rebalance".
3. **Trigger**: Daily, at **3:50 PM Eastern** (20 min before close).
4. **Action**: Start a program.
5. **Program/script**:
   ```
   C:\Users\hassa\Downloads\NueroTrade\.venv\Scripts\python.exe
   ```
6. **Add arguments**:
   ```
   -m scripts.run_csm_live
   ```
7. **Start in**:
   ```
   C:\Users\hassa\Downloads\NueroTrade
   ```
8. Open the task **Properties → Conditions** → uncheck "Start the task only if the computer is on AC power" if you're on a laptop.
9. **Properties → Settings** → check "If the running task does not end when requested, force it to stop" with stop after 10 minutes.

That's it. Verify by:

```
schtasks /query /tn "NeuroTrade CSM Rebalance"
```

## Verify a daily run manually

```cmd
.venv\Scripts\activate
python -m scripts.run_csm_live --dry-run
```

This shows what *would* happen without sending orders. Use it to inspect picks
before any rebalance day.

## Disable / pause

To pause paper trading without deleting state:

```cmd
schtasks /change /tn "NeuroTrade CSM Rebalance" /disable
```

Re-enable with `/enable`.

## Safety guardrails baked into the runner

- **Refuses to run if `ALPACA_PAPER=false`.** Will not touch live capital
  unless you explicitly flip the env var (which you should not, until 60+
  days of clean paper performance).
- **Idempotent**: re-running on the same day is a no-op until the rebalance
  interval elapses.
- **Frozen parameters**: the runner uses the CPCV-winning variant
  (top_n=5, rebalance=15, inverse_vol). Don't tune them without re-running
  `scripts.build_csm_grid` + `scripts.run_cpcv`.

## What to watch on the dashboard

`http://127.0.0.1:8765/` — the Live Bot card will appear after the bot
writes `bot_state.json`. Equity, daily P&L, current positions, and kill
switch status update every 1 second.

## When to scale up (paper → small live → full)

- **First 30 trading days**: just watch. Don't tune. Don't intervene.
  The strategy goes 15 trading days without a rebalance — most days are quiet.
- **Days 30–60**: if Sharpe over the period is > 0.3 and max drawdown
  hasn't exceeded 20%, you've cleared the basic forward-test bar.
- **Days 60–90**: if metrics are still in band, consider live with **5%**
  of intended capital. Re-read [docs/ops_runbook.md](ops_runbook.md) §10.

## What to do if a rebalance fails

The runner exits non-zero on any order failure but does NOT halt the schedule.
Check `logs/csm_live.log` for the specific symbol/error. Common cases:

- `subscription does not permit recent SIP data` — retry after market close.
- `insufficient buying power` — Alpaca paper account has $100k by default;
  this shouldn't happen unless you've manually wound it down.
- `position not found` (during a close) — Alpaca may have already auto-closed
  it from a corporate action. Safe to ignore.

If three consecutive rebalances fail: open `bot_state.json` and set
`enabled: false`, then investigate.
