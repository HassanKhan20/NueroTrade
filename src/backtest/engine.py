"""Event-driven backtest engine.

Side-effect: while running, the engine periodically writes
`reports/backtest_progress.json` so the dashboard can show a live progress
bar and per-day equity for the current run. The file is removed when the
run completes or errors. Writes are atomic (tmp + rename).

Simulates one trading day at a time:
  1. Components emit Trade intents (long-only).
  2. Allocator scales each by component weight × regime multiplier.
  3. Engine walks the day's 1-min bars and fills:
       * ORB: when intraday high crosses the buy-stop trigger price.
       * Intraday momentum: instantly at 15:00 close.
     After fill, stop-loss is monitored bar-by-bar; otherwise force-close at exit_time.
  4. Commissions and slippage applied on entry and exit.
  5. Equity updated daily; kill switch checked.

Inputs are deterministic: identical bars + params -> identical output.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.costs.commissions import commission as _commission
from src.costs.slippage import fill_price as _fill_price
from src.ensemble.allocator import AllocatorParams, apply_allocation
from src.ensemble.kill_switch import KillSwitch, KillSwitchState
from src.strategies.base import Side, Trade
from src.strategies.ict_long_only import ICTParams, ict_signals
from src.strategies.intraday_momentum import (
    IntradayMomentumParams,
    intraday_momentum_signals,
)
from src.strategies.orb_stocks_in_play import ORBParams, orb_signals
from src.strategies.vol_regime import VolRegimeParams, regime_multiplier
from src.utils.clock import NY_TZ, is_trading_day


@dataclass
class FilledTrade:
    """A Trade that was actually filled, with realized P&L."""

    intent: Trade
    fill_price: float
    exit_price: float
    exit_reason: str  # "stop" | "time" | "kill_switch"
    pnl_gross: float
    commission: float

    @property
    def pnl_net(self) -> float:
        return self.pnl_gross - self.commission

    @property
    def symbol(self) -> str:
        return self.intent.symbol

    @property
    def component(self) -> str:
        return self.intent.component


@dataclass
class BacktestResult:
    equity_curve: pd.Series  # daily, indexed by date
    trades: list[FilledTrade]
    daily_returns: pd.Series  # arithmetic
    component_attribution: dict[str, float]  # net pnl per component

    def n_trades(self) -> int:
        return len(self.trades)


DEFAULT_PROGRESS_PATH = (
    Path(__file__).resolve().parents[2] / "reports" / "backtest_progress.json"
)


def _write_progress(path: Path, payload: dict) -> None:
    """Atomic JSON write so the dashboard never reads a partial file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, default=str))
        os.replace(tmp, path)
    except OSError:
        pass  # progress is non-critical; never let it crash the engine


def _clear_progress(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@dataclass
class BacktestConfig:
    initial_equity: float = 25_000.0
    orb_params: ORBParams = field(default_factory=ORBParams)
    momentum_params: IntradayMomentumParams = field(
        default_factory=IntradayMomentumParams
    )
    ict_params: ICTParams = field(default_factory=ICTParams)
    vol_regime_params: VolRegimeParams = field(default_factory=VolRegimeParams)
    allocator_params: AllocatorParams = field(default_factory=AllocatorParams)
    daily_loss_limit: float = 0.025
    weekly_loss_limit: float = 0.05
    enable_intraday_momentum: bool = True
    enable_orb: bool = True
    # Progress reporting (for the dashboard). Set to None to disable.
    progress_path: Path | None = field(default=DEFAULT_PROGRESS_PATH)
    progress_label: str = ""
    # Write progress every N processed days (cheap; default ~once per second).
    progress_every_n_days: int = 1
    # ICT mode: when enabled, the engine ignores ORB + intraday momentum and
    # runs the multi-timeframe-alignment strategy instead. The two cadences
    # don't compose cleanly (ICT fires inside killzones; ORB at 09:35; momentum
    # at 15:00) so they're mutually exclusive.
    enable_ict: bool = False


@dataclass
class DayInputs:
    """All data for one trading day, pre-aligned by the caller."""

    session_date: date
    minute_bars_orb: dict[str, pd.DataFrame]      # symbol -> 1-min bars (NY-tz)
    daily_bars_orb: dict[str, pd.DataFrame]       # symbol -> daily bars history
    minute_bars_mom: dict[str, pd.DataFrame]      # SPY/QQQ 1-min
    prev_close_mom: dict[str, float]              # SPY/QQQ prev close
    vix_today: float
    vix_history: pd.Series
    spy_close_history: pd.Series
    # ICT-only: the QQQ 1-min frame is reused for the index-bias filter.
    # When ICT mode is off, this can be empty.
    qqq_1min: pd.DataFrame = field(default_factory=pd.DataFrame)


def _simulate_orb_trade(
    trade: Trade, day_bars: pd.DataFrame
) -> FilledTrade | None:
    """Walk the day's bars from 09:35 onward; fill when high >= entry_price.

    Clips to the trade's session date so a multi-day bar history doesn't
    leak future bars into the fill simulation.
    """
    if not isinstance(day_bars.index, pd.DatetimeIndex):
        return None
    sess = trade.entry_time.date()
    same_day = day_bars[day_bars.index.date == sess]
    after_or = same_day[same_day.index >= trade.entry_time]
    if after_or.empty:
        return None
    fill_bar = None
    for ts, row in after_or.iterrows():
        if row["high"] >= trade.entry_price:
            fill_bar = (ts, row)
            break
    if fill_bar is None:
        return None  # never broke OR-high — no entry

    fill_ts, fill_row = fill_bar
    entry_fill = _fill_price(
        +1,
        trade.entry_price,
        {"high": float(fill_row["high"]), "low": float(fill_row["low"])},
    )

    # Walk forward checking take-profit (if set), then stop, then time exit.
    # TP takes precedence over stop within the same bar — open-gaps-up
    # scenarios usually fill the TP before the stop is touched.
    after_fill = after_or[after_or.index >= fill_ts]
    exit_price = float("nan")
    exit_reason = "time"
    tp = trade.take_profit_price
    for ts, row in after_fill.iterrows():
        if ts >= trade.exit_time:
            exit_price = _fill_price(
                -1,
                float(row["open"]),
                {"high": float(row["high"]), "low": float(row["low"])},
            )
            exit_reason = "time"
            break
        if tp is not None and row["high"] >= tp:
            exit_price = _fill_price(
                -1,
                tp,
                {"high": float(row["high"]), "low": float(row["low"])},
            )
            exit_reason = "take_profit"
            break
        if row["low"] <= trade.stop_price:
            exit_price = _fill_price(
                -1,
                trade.stop_price,
                {"high": float(row["high"]), "low": float(row["low"])},
            )
            exit_reason = "stop"
            break
    if not np.isfinite(exit_price):
        last_row = after_fill.iloc[-1]
        exit_price = _fill_price(
            -1,
            float(last_row["close"]),
            {"high": float(last_row["high"]), "low": float(last_row["low"])},
        )
        exit_reason = "time"

    pnl = (exit_price - entry_fill) * trade.shares
    comm = _commission(trade.shares) * 2
    return FilledTrade(
        intent=trade,
        fill_price=entry_fill,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl_gross=pnl,
        commission=comm,
    )


def _simulate_momentum_trade(
    trade: Trade, day_bars: pd.DataFrame
) -> FilledTrade | None:
    """Fill at 15:00 close, exit at 15:55 close — both with slippage.

    Clips to the trade's session date so a multi-day bar history doesn't
    leak future bars into the fill simulation.
    """
    if not isinstance(day_bars.index, pd.DatetimeIndex):
        return None
    sess = trade.entry_time.date()
    same_day = day_bars[day_bars.index.date == sess]
    entry_window = same_day[same_day.index >= trade.entry_time]
    if entry_window.empty:
        return None
    entry_bar = entry_window.iloc[0]
    entry_fill = _fill_price(
        +1,
        float(entry_bar["close"]),
        {"high": float(entry_bar["high"]), "low": float(entry_bar["low"])},
    )

    # Walk bar-by-bar to allow take-profit (and notional stop) to fire
    # intraday rather than waiting for time-exit.
    tp = trade.take_profit_price
    exit_price = float("nan")
    exit_reason = "time"
    for ts, row in entry_window.iterrows():
        if ts >= trade.exit_time:
            exit_price = _fill_price(
                -1,
                float(row["open"]) if "open" in row else float(row["close"]),
                {"high": float(row["high"]), "low": float(row["low"])},
            )
            exit_reason = "time"
            break
        if tp is not None and row["high"] >= tp:
            exit_price = _fill_price(
                -1,
                tp,
                {"high": float(row["high"]), "low": float(row["low"])},
            )
            exit_reason = "take_profit"
            break
        if row["low"] <= trade.stop_price:
            exit_price = _fill_price(
                -1,
                trade.stop_price,
                {"high": float(row["high"]), "low": float(row["low"])},
            )
            exit_reason = "stop"
            break
    if not np.isfinite(exit_price):
        last_row = entry_window.iloc[-1]
        exit_price = _fill_price(
            -1,
            float(last_row["close"]),
            {"high": float(last_row["high"]), "low": float(last_row["low"])},
        )
        exit_reason = "time"

    pnl = (exit_price - entry_fill) * trade.shares
    comm = _commission(trade.shares) * 2
    return FilledTrade(
        intent=trade,
        fill_price=entry_fill,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl_gross=pnl,
        commission=comm,
    )


class Backtester:
    def __init__(self, cfg: BacktestConfig | None = None) -> None:
        self.cfg = cfg or BacktestConfig()

    def run(self, days: list[DayInputs]) -> BacktestResult:
        equity = self.cfg.initial_equity
        ks = KillSwitch(
            daily_loss_limit=self.cfg.daily_loss_limit,
            weekly_loss_limit=self.cfg.weekly_loss_limit,
        )
        equity_by_day: dict[date, float] = {}
        all_trades: list[FilledTrade] = []
        attribution: dict[str, float] = {
            "orb": 0.0,
            "intraday_momentum": 0.0,
            "ict": 0.0,
        }
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        n_total = len(days)
        progress_path = self.cfg.progress_path

        def _push_progress(processed: int, current_date: date | None) -> None:
            if progress_path is None:
                return
            payload = {
                "label": self.cfg.progress_label or "backtest",
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "processed": processed,
                "total": n_total,
                "pct_done": (processed / n_total * 100.0) if n_total else 0.0,
                "current_date": current_date.isoformat() if current_date else None,
                "equity": equity,
                "n_trades": len(all_trades),
                "attribution": dict(attribution),
                "regime": "ict" if self.cfg.enable_ict else "ensemble",
            }
            _write_progress(progress_path, payload)

        if progress_path is not None:
            _push_progress(0, days[0].session_date if days else None)

        for i, day in enumerate(days, start=1):
            if not is_trading_day(day.session_date):
                equity_by_day[day.session_date] = equity
                continue

            ks.reset_for_day(day.session_date, equity)
            if not ks.can_trade():
                equity_by_day[day.session_date] = equity
                continue

            mult = regime_multiplier(
                day.vix_today,
                day.vix_history,
                day.spy_close_history,
                self.cfg.vol_regime_params,
            )
            if mult <= 0:
                equity_by_day[day.session_date] = equity
                continue

            intents: list[Trade] = []
            if self.cfg.enable_ict:
                # ICT mode replaces ORB + intraday momentum entirely.
                intents += ict_signals(
                    day.minute_bars_orb,
                    day.daily_bars_orb,
                    day.qqq_1min,
                    day.session_date,
                    equity=equity,
                    params=self.cfg.ict_params,
                )
            else:
                if self.cfg.enable_orb:
                    intents += orb_signals(
                        day.minute_bars_orb,
                        day.daily_bars_orb,
                        day.session_date,
                        equity=equity,
                        params=self.cfg.orb_params,
                    )
                if self.cfg.enable_intraday_momentum:
                    intents += intraday_momentum_signals(
                        day.minute_bars_mom,
                        day.prev_close_mom,
                        day.session_date,
                        equity=equity,
                        params=self.cfg.momentum_params,
                    )

            if self.cfg.enable_ict:
                # ICT runs solo: skip the ensemble 60/30/10 weights, just apply
                # the regime multiplier so vol-spike days size down.
                sized: list[Trade] = []
                for t in intents:
                    scaled = int(t.shares * mult)
                    if scaled <= 0:
                        continue
                    sized.append(
                        Trade(
                            symbol=t.symbol, side=t.side,
                            entry_time=t.entry_time, entry_price=t.entry_price,
                            stop_price=t.stop_price, shares=scaled,
                            exit_time=t.exit_time, component=t.component,
                            take_profit_price=t.take_profit_price, meta=t.meta,
                        )
                    )
            else:
                sized = apply_allocation(
                    intents,
                    regime_multiplier=mult,
                    params=self.cfg.allocator_params,
                )

            day_pnl = 0.0
            for intent in sized:
                if intent.component == "intraday_momentum":
                    day_bars = day.minute_bars_mom.get(intent.symbol)
                else:
                    # orb + ict both come from the per-symbol 1-min cache
                    day_bars = day.minute_bars_orb.get(intent.symbol)
                if day_bars is None or day_bars.empty:
                    continue
                if intent.component == "intraday_momentum":
                    filled = _simulate_momentum_trade(intent, day_bars)
                else:
                    # ORB and ICT share the bracket-style fill simulator:
                    # walk forward, stop / TP / time-exit, all clipped to today.
                    filled = _simulate_orb_trade(intent, day_bars)
                if filled is None:
                    continue
                all_trades.append(filled)
                day_pnl += filled.pnl_net
                attribution[intent.component] += filled.pnl_net
                # Update kill switch mid-day; if breached, halt remaining intents.
                running = equity + day_pnl
                state = ks.update_equity(running)
                if state != KillSwitchState.OK:
                    break

            equity += day_pnl
            equity_by_day[day.session_date] = equity

            if progress_path is not None and (
                i % max(1, self.cfg.progress_every_n_days) == 0 or i == n_total
            ):
                _push_progress(i, day.session_date)

        # Run completed — clear the live progress file.
        if progress_path is not None:
            _clear_progress(progress_path)

        eq_series = pd.Series(equity_by_day, name="equity").sort_index()
        rets = eq_series.pct_change().fillna(0.0)
        return BacktestResult(
            equity_curve=eq_series,
            trades=all_trades,
            daily_returns=rets,
            component_attribution=attribution,
        )
