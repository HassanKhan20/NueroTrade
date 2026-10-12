"""Compute callout-feed events from one screener snapshot to the next.

The screener writes a list of graded callouts. The watcher loop calls
``diff_picks(prev, curr)`` after each scan and gets back a list of events
suitable for appending to a JSONL feed file:

  - NEW            — symbol was not in the previous picks (with grade >= C)
  - UPGRADED       — grade moved up (D->C, C->B, B->A, A->A+)
  - DOWNGRADED     — grade moved down
  - ACTION_CHANGED — action verb changed (e.g. WAIT -> ENTER)
  - DROPPED        — was a pick, no longer is

Pure function; no I/O.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

GRADE_RANK = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4, "A+": 5}


def _by_symbol(picks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(p.get("symbol", "")): p for p in picks if p.get("symbol")}


def _action_verb(action: str | None) -> str:
    if not action:
        return ""
    a = action.upper()
    for verb in ("ENTER", "WAIT", "REDUCE", "SKIP", "WATCHLIST"):
        if a.startswith(verb):
            return verb
    return a.split(" ", 1)[0]


def _event(kind: str, ts: str, p: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Build a feed event with the relevant pick fields plus extras."""
    keep = (
        "symbol", "grade", "composite_score", "entry_price", "stop_price",
        "take_profit_1", "take_profit_2", "reward_to_risk_tp1",
        "reward_to_risk_tp2", "suggested_shares", "notional", "action",
        "atr_14", "rank", "cap_tier", "halal",
    )
    out: dict[str, Any] = {"ts": ts, "event": kind}
    for k in keep:
        if k in p:
            out[k] = p[k]
    out.update(extra)
    return out


def diff_picks(
    prev: list[dict[str, Any]] | None,
    curr: list[dict[str, Any]],
    *,
    min_grade: str = "C",
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return events between previous and current picks lists.

    Only symbols whose grade is >= ``min_grade`` (in the current OR previous
    snapshot) are considered for events — F/D-grade churn is not interesting.
    """
    ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    prev_map = _by_symbol(prev or [])
    curr_map = _by_symbol(curr or [])
    floor = GRADE_RANK.get(min_grade, 2)
    events: list[dict[str, Any]] = []

    # NEW + UPGRADED + DOWNGRADED + ACTION_CHANGED
    for sym, p in curr_map.items():
        cur_g = GRADE_RANK.get(str(p.get("grade", "F")), 0)
        prv = prev_map.get(sym)
        if prv is None:
            if cur_g >= floor:
                events.append(_event("NEW", ts, p))
            continue
        prv_g = GRADE_RANK.get(str(prv.get("grade", "F")), 0)
        if cur_g > prv_g and (cur_g >= floor or prv_g >= floor):
            events.append(_event(
                "UPGRADED", ts, p,
                prev_grade=prv.get("grade"),
            ))
        elif cur_g < prv_g and (cur_g >= floor or prv_g >= floor):
            events.append(_event(
                "DOWNGRADED", ts, p,
                prev_grade=prv.get("grade"),
            ))
        # Action verb change is independent of grade movement (e.g.,
        # WAIT -> ENTER when RSI cools off without a grade bump).
        prv_verb = _action_verb(prv.get("action"))
        cur_verb = _action_verb(p.get("action"))
        if (prv_verb != cur_verb and cur_g >= floor
                and not any(e["event"] == "UPGRADED" and e["symbol"] == sym
                            for e in events)):
            events.append(_event(
                "ACTION_CHANGED", ts, p,
                prev_action=prv.get("action"),
            ))

    # DROPPED — was a pick at >= min_grade, no longer in current picks
    for sym, prv in prev_map.items():
        if sym in curr_map:
            continue
        prv_g = GRADE_RANK.get(str(prv.get("grade", "F")), 0)
        if prv_g >= floor:
            events.append(_event(
                "DROPPED", ts, prv,
                prev_grade=prv.get("grade"),
            ))

    return events
