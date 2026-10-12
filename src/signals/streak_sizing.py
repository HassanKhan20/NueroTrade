"""Win-streak (anti-martingale) position sizing for the Williams strategy.

Larry Williams' money management: press your bets when you're winning,
pull back to base after a loss. After N consecutive winning callouts the
per-trade risk grows by ``step`` each win, capped at ``cap_pct`` so a hot
streak can never blow the account.

  risk_pct = min(base_pct + win_streak * step, cap_pct)

A LOSS resets the streak to 0 -> back to base risk on the next trade.

HONEST WARNING: this amplifies drawdowns as hard as it amplifies gains.
The cap exists for a reason. Williams blew accounts doing this uncapped.

Win/loss is read from the callouts feed:
  - TP1_HIT or TP2_HIT  -> win
  - STOP_HIT            -> loss
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_RISK_PCT = 0.01      # 1% of equity at streak 0
STEP_PCT = 0.005          # +0.5% risk per consecutive win
CAP_PCT = 0.03            # never risk more than 3% on one trade

WIN_EVENTS = ("TP1_HIT", "TP2_HIT")
LOSS_EVENTS = ("STOP_HIT",)


def streak_risk_pct(
    win_streak: int,
    base_pct: float = BASE_RISK_PCT,
    step: float = STEP_PCT,
    cap_pct: float = CAP_PCT,
) -> float:
    """Risk fraction for the next trade given the current win streak."""
    streak = max(0, int(win_streak))
    return min(base_pct + streak * step, cap_pct)


def current_win_streak(feed_path: Path) -> int:
    """Count consecutive WINS from the most recent resolved callout.

    Walks terminal events (TP/STOP) newest -> oldest. Each TP hit for a
    distinct callout extends the streak; the first STOP hit ends it.

    TP1_HIT followed by TP2_HIT for the SAME callout counts as one win
    (dedup by (symbol, entry_ts)). Returns 0 if no resolved trades.
    """
    if not feed_path.exists():
        return 0
    events: list[dict] = []
    try:
        with feed_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return 0

    # Collapse to one outcome per callout (symbol, entry_ts): a STOP_HIT is
    # decisive (loss); otherwise the best TP reached is the outcome (win).
    outcomes: dict[tuple, dict] = {}  # key -> {ts, result}
    for ev in events:
        kind = ev.get("event")
        if kind not in WIN_EVENTS and kind not in LOSS_EVENTS:
            continue
        sym = ev.get("symbol")
        entry_ts = ev.get("entry_ts") or ev.get("ts")
        key = (sym, entry_ts)
        ts = ev.get("ts", "")
        if kind in LOSS_EVENTS:
            outcomes[key] = {"ts": ts, "result": "loss"}
        else:
            # win — but don't override a recorded loss for the same callout
            prior = outcomes.get(key)
            if prior is None or prior["result"] != "loss":
                outcomes[key] = {"ts": ts, "result": "win"}

    if not outcomes:
        return 0
    # Order by resolution time, newest first; count leading wins.
    ordered = sorted(outcomes.values(), key=lambda o: o["ts"], reverse=True)
    streak = 0
    for o in ordered:
        if o["result"] == "win":
            streak += 1
        else:
            break
    return streak
