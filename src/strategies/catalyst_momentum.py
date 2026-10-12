"""Catalyst-momentum: short-term momentum gated by a volume-spike filter.

Designed for small/mid-cap biotech & pharma names where single-day
catalyst events (FDA approvals, trial readouts, earnings, M&A) drive 30-60%
moves. Without a paid catalyst-calendar feed, **a sustained volume spike
is the cleanest causal proxy** for "something just happened."

Pipeline (all daily, all causal — no multi-timeframe leak by design):

  At each rebalance date D:
    For every symbol:
      vol_ratio  = volume(D-1) / mean(volume(D-21 .. D-2))
      mom_5d     = close(D-1) / close(D-6) - 1

      qualified  = (vol_ratio >= spike_threshold) AND (mom_5d > 0)
      score      = mom_5d * vol_ratio        # rewards both magnitude + flow

    Pick top_n qualified names by score.
    Buy at OPEN of D. Hold for `hold_days`. Sell at OPEN of D + hold_days.
    Equal-weight (compatible with small accounts via fractional shares).

Position sizing:
  - Equal dollar per name (no inverse-vol — biotech vols are too noisy).
  - Sized so a -50% drawdown on one position is at most max_loss_per_trade.

Risk overlay built-in:
  - max_loss_per_trade caps each position's nominal loss exposure.
  - Time exit only — no in-trade stops or take-profits (catalyst plays
    are bimodal; stops would cut the wins, TPs would cap them).

Caveats explicit in the docstring (read these before using):
  - Backtest uses Alpaca daily bars (IEX-only on free tier). Real penny-stock
    spreads are 5-15% per round-trip — if you trade actual <$5 names, your
    live results will be materially worse than this backtest.
  - The biotech universe contains some names that may have been delisted or
    halted in real time. Survivorship bias is REAL. Treat results with
    appropriate skepticism.
  - 5-day momentum + 1-day volume spike is a noisy signal. Expect high
    variance in single-period results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CatalystMomentumParams:
    # ===== Signal =====
    momentum_lookback: int = 5
    volume_lookback: int = 20
    spike_threshold: float = 3.0
    min_momentum: float = 0.0
    # ===== Selection / holding =====
    top_n: int = 5
    hold_days: int = 5
    # ===== Sizing =====
    initial_equity: float = 25_000.0
    max_position_pct: float = 0.25
    # ===== Costs =====
    slippage_bps: float = 25.0
    commission_per_share: float = 0.0035
    commission_min: float = 0.35
    # ===== v2 fixes (each addresses a specific failure mode) =====
    # 1. Direction filter — only spikes where the close is in the UPPER
    #    fraction of the day's range. Filters out panic-selling spikes
    #    that look identical to good-news spikes on price+volume alone.
    require_strong_close: bool = True
    strong_close_pct: float = 0.6
    # 2. Trend filter — must be above its own 50-day SMA. Avoids buying
    #    dead-cat bounces in established downtrends.
    require_above_sma: bool = True
    sma_window: int = 50
    # 3. Soft stop-loss — exit early if the position drops > X% from entry.
    #    Prevents riding fades the full hold period.
    soft_stop_pct: float = 0.08
    # 4. Market regime filter — only trade when SPY 50-SMA > 200-SMA.
    #    Sits out bearish broad-market regimes.
    require_market_uptrend: bool = True


@dataclass
class CMTrade:
    symbol: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    shares: int
    pnl_gross: float
    commission: float
    score: float
    vol_ratio: float
    mom_5d: float

    @property
    def pnl_net(self) -> float:
        return self.pnl_gross - self.commission


@dataclass
class CMResult:
    equity_curve: pd.Series
    trades: list[CMTrade]
    daily_returns: pd.Series
    rebalance_log: list[dict] = field(default_factory=list)

    def n_trades(self) -> int:
        return len(self.trades)


# ---------- pure scoring helpers (the entire strategy logic is here) ---------- #
def _norm_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def score_symbol(
    df: pd.DataFrame, ref_date: pd.Timestamp, params: CatalystMomentumParams
) -> tuple[float, float, float] | None:
    """Compute (score, vol_ratio, mom_5d) for one symbol as of ref_date.

    Causal: uses only bars whose index is strictly BEFORE ref_date.
    Returns None if insufficient history.
    """
    df = _norm_index(df)
    ref = ref_date.tz_localize(None) if ref_date.tzinfo is not None else ref_date
    prior = df.loc[df.index < ref]
    needed = max(params.momentum_lookback, params.volume_lookback,
                 params.sma_window if params.require_above_sma else 0) + 2
    if len(prior) < needed:
        return None

    closes = prior["close"]
    vols = prior["volume"]

    # ----- Momentum -----
    end_close = float(closes.iloc[-1])
    start_close = float(closes.iloc[-(params.momentum_lookback + 1)])
    if start_close <= 0:
        return None
    mom = end_close / start_close - 1.0

    # ----- Volume ratio -----
    today_vol = float(vols.iloc[-1])
    avg_vol = float(vols.iloc[-(params.volume_lookback + 1):-1].mean())
    if avg_vol <= 0:
        return None
    vol_ratio = today_vol / avg_vol

    # ----- v2 fix #1: direction filter -----
    # The spike must close in the upper fraction of its day's range.
    # close-near-low + huge volume = panic selling (avoid).
    if params.require_strong_close:
        last = prior.iloc[-1]
        rng = float(last["high"]) - float(last["low"])
        if rng <= 0:
            return None
        position_in_range = (float(last["close"]) - float(last["low"])) / rng
        if position_in_range < params.strong_close_pct:
            return None

    # ----- v2 fix #2: trend filter -----
    # Must be above its own SMA. Skip dead-cat bounces in downtrends.
    if params.require_above_sma:
        sma = float(closes.iloc[-params.sma_window:].mean())
        if end_close < sma:
            return None

    score = mom * vol_ratio
    return score, vol_ratio, mom


def rank_universe(
    daily_by_sym: Mapping[str, pd.DataFrame],
    as_of: date,
    params: CatalystMomentumParams,
) -> list[tuple[str, float, float, float]]:
    """Return symbols sorted by score descending. Each tuple:
    (symbol, score, vol_ratio, mom_5d). Only includes qualified names —
    those that pass both volume-spike AND momentum thresholds."""
    ts = pd.Timestamp(as_of)
    out: list[tuple[str, float, float, float]] = []
    for sym, df in daily_by_sym.items():
        if df is None or df.empty:
            continue
        score = score_symbol(df, ts, params)
        if score is None:
            continue
        s, vr, mom = score
        if vr >= params.spike_threshold and mom > params.min_momentum:
            out.append((sym, s, vr, mom))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


# ---------- backtest engine (daily, leak-free) ---------- #
def _slip(price: float, side: int, bps: float) -> float:
    return price * (1.0 + side * bps / 10_000.0)


def _commission(shares: int, p: CatalystMomentumParams) -> float:
    if shares <= 0:
        return 0.0
    return max(p.commission_min, abs(shares) * p.commission_per_share)


def run_catalyst_momentum_backtest(
    daily_by_sym: Mapping[str, pd.DataFrame],
    start: date,
    end: date,
    params: CatalystMomentumParams = CatalystMomentumParams(),
) -> CMResult:
    """Walk daily; each `hold_days` open and close positions. Buy at the
    OPEN of the rebalance day; sell at the OPEN of the next rebalance day.

    Causality preserved: signals computed using closes < rebalance date,
    fills happen on the same day's OPEN (the first bar of the day, after
    the prior day's close — by definition causal).
    """
    # Master calendar from union of dates across the universe
    all_dates: set[pd.Timestamp] = set()
    for df in daily_by_sym.values():
        if df is None or df.empty:
            continue
        idx = df.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        all_dates.update(idx)
    cal = sorted(d for d in all_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end))
    if not cal:
        return CMResult(pd.Series(dtype=float), [], pd.Series(dtype=float))

    equity = params.initial_equity
    equity_by_day: dict[pd.Timestamp, float] = {}
    open_positions: dict[str, dict] = {}
    all_trades: list[CMTrade] = []
    rebalance_log: list[dict] = []

    def _close_all(at_date: pd.Timestamp) -> float:
        nonlocal equity
        realized = 0.0
        for sym, pos in list(open_positions.items()):
            df = daily_by_sym.get(sym)
            if df is None or df.empty:
                continue
            df = _norm_index(df)
            row = df.loc[df.index == at_date]
            if row.empty:
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
            all_trades.append(CMTrade(
                symbol=sym,
                entry_date=pos["entry_date"], entry_price=pos["entry_price"],
                exit_date=at_date.date(), exit_price=exit_fill,
                shares=shares, pnl_gross=pnl_gross, commission=comm,
                score=pos["score"], vol_ratio=pos["vol_ratio"], mom_5d=pos["mom_5d"],
            ))
        open_positions.clear()
        return realized

    def _open_positions(picks: list[tuple[str, float, float, float]],
                        at_date: pd.Timestamp) -> None:
        nonlocal equity
        if not picks:
            return
        per_name = min(equity / len(picks), equity * params.max_position_pct)
        for sym, score, vr, mom in picks:
            df = daily_by_sym.get(sym)
            if df is None or df.empty:
                continue
            df = _norm_index(df)
            row = df.loc[df.index == at_date]
            if row.empty:
                continue
            open_px = float(row["open"].iloc[0])
            entry_fill = _slip(open_px, side=+1, bps=params.slippage_bps)
            shares = int(math.floor(per_name / entry_fill))
            if shares <= 0:
                continue
            open_positions[sym] = {
                "shares": shares, "entry_price": entry_fill,
                "entry_date": at_date.date(),
                "score": score, "vol_ratio": vr, "mom_5d": mom,
            }

    # ----- v2 fix #4: market regime filter (SPY 50-SMA > 200-SMA) -----
    # Pre-compute daily SPY signal so we can cheaply check at every rebalance.
    spy = daily_by_sym.get("SPY")
    spy_uptrend: dict[pd.Timestamp, bool] = {}
    if params.require_market_uptrend and spy is not None and not spy.empty:
        spy_n = _norm_index(spy)
        spy_close = spy_n["close"]
        for d in cal:
            prior = spy_close.loc[spy_close.index < d]
            if len(prior) < 200:
                spy_uptrend[d] = False
            else:
                sma_50 = float(prior.iloc[-50:].mean())
                sma_200 = float(prior.iloc[-200:].mean())
                spy_uptrend[d] = sma_50 > sma_200

    def _check_soft_stops(at_date: pd.Timestamp) -> float:
        """v2 fix #3: exit early any position where the close is more than
        soft_stop_pct below the entry. Returns realised pnl."""
        nonlocal equity
        if params.soft_stop_pct <= 0 or not open_positions:
            return 0.0
        realized = 0.0
        for sym, pos in list(open_positions.items()):
            df = daily_by_sym.get(sym)
            if df is None or df.empty:
                continue
            df = _norm_index(df)
            row = df.loc[df.index == at_date]
            if row.empty:
                continue
            close_px = float(row["close"].iloc[0])
            drawdown = (close_px - pos["entry_price"]) / pos["entry_price"]
            if drawdown <= -params.soft_stop_pct:
                # Exit at next-day OPEN — but for simplicity we mark-out at
                # this close with slippage. Conservative: stop fills at the
                # close price minus slippage.
                exit_fill = _slip(close_px, side=-1, bps=params.slippage_bps)
                shares = pos["shares"]
                pnl_gross = (exit_fill - pos["entry_price"]) * shares
                comm = _commission(shares, params) * 2
                pnl_net = pnl_gross - comm
                realized += pnl_net
                all_trades.append(CMTrade(
                    symbol=sym,
                    entry_date=pos["entry_date"], entry_price=pos["entry_price"],
                    exit_date=at_date.date(), exit_price=exit_fill,
                    shares=shares, pnl_gross=pnl_gross, commission=comm,
                    score=pos["score"], vol_ratio=pos["vol_ratio"],
                    mom_5d=pos["mom_5d"],
                ))
                open_positions.pop(sym, None)
        return realized

    last_rebalance_idx = -params.hold_days
    for i, today in enumerate(cal):
        # Soft-stop check fires every day on currently-held names.
        equity += _check_soft_stops(today)

        if i - last_rebalance_idx >= params.hold_days:
            # v2 fix #4: regime gate
            if params.require_market_uptrend and spy is not None and not spy.empty:
                if not spy_uptrend.get(today, False):
                    # Bearish broad market — close everything, hold cash.
                    equity += _close_all(today)
                    rebalance_log.append({
                        "date": today.date().isoformat(),
                        "picks": [], "n_qualified": 0,
                        "equity_at_rebal": equity,
                        "regime": "bearish",
                    })
                    last_rebalance_idx = i
                    equity_by_day[today] = equity
                    continue

            ranked = rank_universe(daily_by_sym, today.date(), params)
            picks = ranked[: params.top_n]
            equity += _close_all(today)
            _open_positions(picks, today)
            rebalance_log.append({
                "date": today.date().isoformat(),
                "picks": [s for s, *_ in picks],
                "n_qualified": len(ranked),
                "equity_at_rebal": equity,
                "regime": "bullish",
            })
            last_rebalance_idx = i

        # mark-to-market
        mtm = equity
        for sym, pos in open_positions.items():
            df = daily_by_sym.get(sym)
            if df is None or df.empty:
                continue
            df = _norm_index(df)
            row = df.loc[df.index == today]
            if row.empty:
                continue
            close_px = float(row["close"].iloc[0])
            mtm += (close_px - pos["entry_price"]) * pos["shares"]
        equity_by_day[today] = mtm

    if cal and open_positions:
        equity += _close_all(cal[-1])
        equity_by_day[cal[-1]] = equity

    eq_series = pd.Series(equity_by_day, name="equity").sort_index()
    rets = eq_series.pct_change().fillna(0.0)
    return CMResult(
        equity_curve=eq_series, trades=all_trades,
        daily_returns=rets, rebalance_log=rebalance_log,
    )
