"""bot_state.json reader/writer.

Single source of truth that the dashboard polls. All fields are JSON-safe.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BotState:
    enabled: bool = True
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    updated_at: str = ""
    equity: float = 0.0
    cash: float = 0.0
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    today_pnl: float = 0.0
    today_trades: int = 0
    kill_switch_state: str = "ok"
    regime_multiplier: float = 1.0
    component_attribution: dict[str, float] = field(
        default_factory=lambda: {"orb": 0.0, "intraday_momentum": 0.0}
    )
    last_event: str = ""

    def stamp(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_state(state: BotState, path: Path) -> None:
    state.stamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2, default=str))
    tmp.replace(path)


def read_state(path: Path) -> BotState | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return BotState(**data)
