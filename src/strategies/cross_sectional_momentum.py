"""Cross-sectional momentum (Jegadeesh-Titman 1993; AQR replication).

Long-only daily-timeframe strategy. Rank a fixed universe of symbols by
their 12-month-minus-1-month return, hold the top-N for one rebalance
interval, replace the bottom-rank names on the schedule, repeat.

Why this specific strategy:
  * Daily bars only -> no multi-timeframe resample -> no leak risk
    by construction.
  * 30+ years of academic evidence (JT 1993, Carhart 1997, AQR replications
    across global markets).
  * Long-only friendly: just buy the winners.
  * Trivial to implement, trivial to verify by hand.
  * Slippage-tolerant: monthly turnover is low.

Signal:
  r_12_1(t) = close(t - skip) / close(t - skip - lookback) - 1
  where lookback ~ 252 trading days, skip ~ 21 trading days.
  The skip excludes the most recent month to avoid short-term reversal
  effects (Jegadeesh 1990).

Pipeline:
  At each rebalance date:
    - Compute r_12_1 for every symbol with enough history.
    - Sort descending.
    - Buy top_n names equal-weight, replacing prior-period holdings.
    - Hold until the next rebalance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CSMParams:
    lookback_days: int = 252
    skip_days: int = 21
    top_n: int = 5
    rebalance_days: int = 21
    target_risk_per_trade: float = 0.005
    initial_equity: float = 25_000.0
    commission_per_share: float = 0.0035
    commission_min: float = 0.35
    slippage_bps: float = 5.0
    # v2 recalibrations (added after the 2023-2024 failure analysis):
    # Time-series momentum filter (Moskowitz-Ooi-Pedersen 2012). Only HOLD
    # names whose own r_12_1 exceeds this. Default 0.0 = "must be positive
    # in absolute terms." Strategy goes to cash in regimes where nothing
    # qualifies.
    abs_momentum_threshold: float = 0.0
    # Weight scheme: "equal" (v1 default) or "inverse_vol".
    # Inverse-vol caps the damage from a single high-vol winner crashing.
    weight_scheme: str = "inverse_vol"
    vol_lookback_days: int = 60


@dataclass
class CSMTrade:
    symbol: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    shares: int
    pnl_gross: float
    commission: float

    @property
    def pnl_net(self) -> float:
        return self.pnl_gross - self.commission


@dataclass
class CSMResult:
    equity_curve: pd.Series        # daily, indexed by date
    trades: list[CSMTrade]
    daily_returns: pd.Series
    rebalance_log: list[dict] = field(default_factory=list)

    def n_trades(self) -> int:
        return len(self.trades)


def _slip(price: float, side: int, bps: float) -> float:
    return price * (1.0 + side * bps / 10_000.0)


def _commission(shares: int, p: CSMParams) -> float:
    if shares <= 0:
        return 0.0
    return max(p.commission_min, abs(shares) * p.commission_per_share)


def _r_12_1(closes: pd.Series, ref_date: pd.Timestamp,
            lookback: int, skip: int) -> float:
    """Compute (close at ref-skip) / (close at ref-skip-lookback) - 1.

    Returns nan if either reference index isn't available.
    """
    prior = closes.loc[closes.index < ref_date]
    if len(prior) < lookback + skip + 1:
        return float("nan")
    end_close = float(prior.iloc[-skip - 1])
    start_close = float(prior.iloc[-skip - lookback - 1])
    if start_close <= 0:
        return float("nan")
    return end_close / start_close - 1.0


def rank_universe(
    daily_by_sym: Mapping[str, pd.DataFrame],
    as_of: date,
    params: CSMParams,
) -> list[tuple[str, float]]:
    """Return symbols sorted by r_12_1 descending. Symbols with insufficient
    history are dropped."""
    ts = pd.Timestamp(as_of)
    scored: list[tuple[str, float]] = []
    for sym, df in daily_by_sym.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        ref = ts.tz_localize(None) if ts.tzinfo is not None else ts
        r = _r_12_1(df["close"], ref, params.lookback_days, params.skip_days)
        if math.isnan(r):
            continue
        scored.append((sym, r))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def run_csm_backtest(
    daily_by_sym: Mapping[str, pd.DataFrame],
    start: date,
    end: date,
    params: CSMParams = CSMParams(),
) -> CSMResult:
    """Walk daily, rebalance every `rebalance_days` trading days. Buy at
    next-day open, exit at next rebalance's open.

    Causality: signals computed using closes strictly before `as_of`. Fills
    happen on the open of `as_of`. No look-ahead.
    """
    # Build a master calendar from the union of dates in our daily frames
    all_dates: set[pd.Timestamp] = set()
    for df in daily_by_sym.values():
        if df is None or df.empty:
            continue
        idx = df.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        all_dates.update(idx)
    cal = sorted(all_dates)
    cal = [d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not cal:
        return CSMResult(
            equity_curve=pd.Series(dtype=float),
            trades=[], daily_returns=pd.Series(dtype=float),
        )

    equity = params.initial_equity
    equity_by_day: dict[pd.Timestamp, float] = {}
    open_trades: dict[str, dict] = {}  # symbol -> {shares, entry_price, entry_date}
    all_trades: list[CSMTrade] = []
    rebalance_log: list[dict] = []

    def _close_all(at_date: pd.Timestamp) -> float:
        """Close every open trade at this date's OPEN price; return realized pnl."""
        nonlocal equity
        realized = 0.0
        for sym, pos in list(open_trades.items()):
            df = daily_by_sym.get(sym)
            if df is None or df.empty:
                continue
            idx = df.index
            if idx.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
            row = df.loc[df.index == at_date]
            if row.empty:
                # symbol has no bar today -> use last available close
                prior = df.loc[df.index <= at_date]
                if prior.empty:
                    continue
                exit_px = float(prior["close"].iloc[-1])
            else:
                exit_px = float(row["open"].iloc[0])
            exit_fill = _slip(exit_px, side=-1, bps=params.slippage_bps)
            shares = pos["shares"]
            pnl_gross = (exit_fill - pos["entry_price"]) * shares
            comm = _commission(shares, params) * 2  # round-trip
            pnl_net = pnl_gross - comm
            realized += pnl_net
            all_trades.append(CSMTrade(
                symbol=sym, entry_date=pos["entry_date"], entry_price=pos["entry_price"],
                exit_date=at_date.date(), exit_price=exit_fill,
                shares=shares, pnl_gross=pnl_gross, commission=comm,
            ))
        open_trades.clear()
        return realized

    def _realized_vol(sym: str, at_date: pd.Timestamp, n: int) -> float:
        """60-day realized vol of daily log returns, ending strictly BEFORE at_date.

        Returns nan if insufficient history. Used for inverse-vol weighting.
        """
        df = daily_by_sym.get(sym)
        if df is None or df.empty:
            return float("nan")
        idx = df.index
        if idx.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        prior = df.loc[df.index < at_date]
        if len(prior) < n + 1:
            return float("nan")
        closes = prior["close"].iloc[-(n + 1):].to_numpy(dtype=float)
        if (closes <= 0).any():
            return float("nan")
        rets = np.diff(np.log(closes))
        sd = float(np.std(rets, ddof=1))
        return sd if sd > 1e-9 else float("nan")

    def _weights_for(picks: list[str], at_date: pd.Timestamp) -> dict[str, float]:
        """Allocate equity proportionally given the configured weight scheme.

        Returns symbol -> dollar allocation. Sum equals current equity.
        """
        if not picks:
            return {}
        if params.weight_scheme == "equal":
            share = equity / len(picks)
            return {s: share for s in picks}
        # inverse_vol
        invs: dict[str, float] = {}
        for s in picks:
            v = _realized_vol(s, at_date, params.vol_lookback_days)
            if math.isnan(v):
                invs[s] = 0.0
            else:
                invs[s] = 1.0 / v
        total = sum(invs.values())
        if total <= 0:
            # Fall back to equal-weight if no vols are computable
            share = equity / len(picks)
            return {s: share for s in picks}
        return {s: equity * (w / total) for s, w in invs.items()}

    def _open_positions(picks: list[str], at_date: pd.Timestamp) -> None:
        """Open long positions sized per the configured weight scheme."""
        nonlocal equity
        if not picks:
            return
        allocation = _weights_for(picks, at_date)
        for sym in picks:
            per_name = allocation.get(sym, 0.0)
            if per_name <= 0:
                continue
            df = daily_by_sym.get(sym)
            if df is None or df.empty:
                continue
            idx = df.index
            if idx.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
            row = df.loc[df.index == at_date]
            if row.empty:
                continue
            open_px = float(row["open"].iloc[0])
            entry_fill = _slip(open_px, side=+1, bps=params.slippage_bps)
            shares = int(math.floor(per_name / entry_fill))
            if shares <= 0:
                continue
            open_trades[sym] = {
                "shares": shares,
                "entry_price": entry_fill,
                "entry_date": at_date.date(),
            }

    last_rebalance_idx = -params.rebalance_days  # force rebalance on first day
    for i, today in enumerate(cal):
        # Rebalance check (signal computed BEFORE today using closes < today)
        if i - last_rebalance_idx >= params.rebalance_days:
            ranked = rank_universe(daily_by_sym, today.date(), params)
            # v2: time-series filter — only HOLD names whose own r_12_1
            # exceeds the absolute-momentum threshold. Goes to cash if
            # nothing qualifies (Moskowitz-Ooi-Pedersen 2012 overlay).
            qualified = [(s, r) for s, r in ranked if r > params.abs_momentum_threshold]
            picks = [sym for sym, _ in qualified[: params.top_n]]
            equity += _close_all(today)
            _open_positions(picks, today)
            rebalance_log.append({
                "date": today.date().isoformat(),
                "picks": picks,
                "n_ranked": len(ranked),
                "n_qualified": len(qualified),
                "equity_at_rebal": equity,
            })
            last_rebalance_idx = i

        # Mark-to-market for the day's equity (not realized — just for the curve)
        mtm = equity
        for sym, pos in open_trades.items():
            df = daily_by_sym.get(sym)
            if df is None or df.empty:
                continue
            idx = df.index
            if idx.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
            row = df.loc[df.index == today]
            if row.empty:
                continue
            close_px = float(row["close"].iloc[0])
            unreal = (close_px - pos["entry_price"]) * pos["shares"]
            mtm += unreal
        equity_by_day[today] = mtm

    # Final close at end-of-window using last available date
    if cal and open_trades:
        equity += _close_all(cal[-1])
        equity_by_day[cal[-1]] = equity

    eq_series = pd.Series(equity_by_day, name="equity").sort_index()
    rets = eq_series.pct_change().fillna(0.0)
    return CSMResult(
        equity_curve=eq_series,
        trades=all_trades,
        daily_returns=rets,
        rebalance_log=rebalance_log,
    )
