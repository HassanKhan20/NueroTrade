"""Multi-factor long-only swing strategy.

Combines four independent signals into a composite score per name per day:

    1. Technical confluence  (RSI + MACD + SMA filter + relative volume)
    2. Cross-sectional 12-1 momentum
    3. Quality (hand-curated overrides + price-stability fallback)
    4. PEAD score (zero unless an earnings feed is wired in)

The composite is z-scored across the universe at each rebalance date and
the top-N qualified names are bought equal-weight. Holds for ~10 trading
days, replaces names that drop out of the top-N at the next rebalance.

Designed to be CAUSAL by construction (only daily-close data, no
multi-timeframe resampling). All four signals individually use only
bars whose date is strictly before the rebalance date.

Risk overlays:
  - Halal screen runs first — non-compliant names are dropped from
    the candidate universe before scoring.
  - Per-trade soft stop at -8% from entry.
  - Position size: equal dollar across picks, capped at max_position_pct.

This is the strategy described in the deep-research report (Part 9).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from src.signals.momentum import momentum_score
from src.signals.pead import EarningsEvent, pead_score
from src.signals.quality import quality_score
from src.signals.technical import technical_snapshot


@dataclass(frozen=True)
class MultiFactorParams:
    # Signal weights — must sum to 1.0 (will be re-normalised if not).
    weight_technical: float = 0.40
    weight_momentum: float = 0.30
    weight_quality: float = 0.20
    weight_pead: float = 0.10

    # Selection
    top_n: int = 5
    min_composite_score: float = 0.0   # only buy names whose score > this

    # Holding
    rebalance_days: int = 10           # ~2 trading weeks

    # Sizing
    initial_equity: float = 25_000.0
    max_position_pct: float = 0.25
    soft_stop_pct: float = 0.08

    # Costs
    slippage_bps: float = 5.0
    commission_per_share: float = 0.0035
    commission_min: float = 0.35

    # Toggles for ablation studies
    enable_technical: bool = True
    enable_momentum: bool = True
    enable_quality: bool = True
    enable_pead: bool = True

    # Correlation cap — at most this many picks from the same industry
    # (industry, not sector, because "Technology" is too coarse).
    # 0 disables the cap (legacy behaviour).
    max_per_industry: int = 2


@dataclass
class MFSTrade:
    symbol: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    shares: int
    pnl_gross: float
    commission: float
    composite_score: float
    technical_score: float
    momentum_score: float
    quality_score: float
    pead_score: float

    @property
    def pnl_net(self) -> float:
        return self.pnl_gross - self.commission


@dataclass
class MFSResult:
    equity_curve: pd.Series
    trades: list[MFSTrade]
    daily_returns: pd.Series
    rebalance_log: list[dict] = field(default_factory=list)

    def n_trades(self) -> int:
        return len(self.trades)


# ---------- helpers ---------- #
def _norm(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def _slip(price: float, side: int, bps: float) -> float:
    return price * (1.0 + side * bps / 10_000.0)


def _commission(shares: int, p: MultiFactorParams) -> float:
    if shares <= 0:
        return 0.0
    return max(p.commission_min, abs(shares) * p.commission_per_share)


def _zscore(values: dict[str, float]) -> dict[str, float]:
    """Z-score across the universe; NaN values propagate as NaN."""
    arr = np.array([v for v in values.values() if np.isfinite(v)], dtype=float)
    if len(arr) < 2:
        return {k: 0.0 for k in values}
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd <= 0:
        return {k: 0.0 for k in values}
    return {k: ((v - mu) / sd if np.isfinite(v) else 0.0) for k, v in values.items()}


def composite_score_for(
    symbol: str,
    bars: pd.DataFrame,
    ref_date: pd.Timestamp,
    earnings_events: list[EarningsEvent] | None,
    params: MultiFactorParams,
) -> dict[str, float] | None:
    """Compute raw individual scores for one symbol. Returns None if
    insufficient history; otherwise a dict with keys {tech, mom, qual, pead}.
    The combiner z-scores across the universe AFTER this step."""
    snap = technical_snapshot(bars, ref_date) if params.enable_technical else None
    if params.enable_technical and snap is None:
        return None

    tech_raw = float(snap.composite_score()) if snap is not None else 0.0
    mom_raw = momentum_score(bars, ref_date) if params.enable_momentum else 0.0
    qual_raw = quality_score(symbol, bars, ref_date) if params.enable_quality else 0.0
    pead_raw = (
        pead_score(symbol, ref_date.date() if hasattr(ref_date, "date") else ref_date,
                    earnings_events)
        if params.enable_pead
        else 0.0
    )

    if not np.isfinite(tech_raw):
        tech_raw = 0.0
    if not np.isfinite(mom_raw):
        mom_raw = 0.0
    if not np.isfinite(qual_raw):
        qual_raw = 0.0
    if not np.isfinite(pead_raw):
        pead_raw = 0.0

    return {"tech": tech_raw, "mom": mom_raw, "qual": qual_raw, "pead": pead_raw}


def rank_universe(
    daily_by_sym: Mapping[str, pd.DataFrame],
    as_of: date,
    earnings_events: list[EarningsEvent] | None,
    params: MultiFactorParams,
) -> list[tuple[str, float, dict[str, float]]]:
    """Return symbols sorted by composite z-score descending.

    Each element: (symbol, composite_score, raw_scores_dict).
    Names with insufficient history are dropped.
    """
    ref = pd.Timestamp(as_of)
    raw_by_sym: dict[str, dict[str, float]] = {}
    for sym, bars in daily_by_sym.items():
        if bars is None or bars.empty:
            continue
        scores = composite_score_for(sym, bars, ref, earnings_events, params)
        if scores is not None:
            raw_by_sym[sym] = scores
    if not raw_by_sym:
        return []

    # Z-score each component across the universe.
    tech_z = _zscore({s: r["tech"] for s, r in raw_by_sym.items()})
    mom_z = _zscore({s: r["mom"] for s, r in raw_by_sym.items()})
    qual_z = _zscore({s: r["qual"] for s, r in raw_by_sym.items()})
    pead_z = _zscore({s: r["pead"] for s, r in raw_by_sym.items()})

    # Re-normalise weights in case they don't sum to 1.
    w = np.array([params.weight_technical, params.weight_momentum,
                   params.weight_quality, params.weight_pead], dtype=float)
    s = w.sum()
    if s <= 0:
        return []
    w = w / s

    out: list[tuple[str, float, dict[str, float]]] = []
    for sym in raw_by_sym:
        composite = (
            w[0] * tech_z[sym]
            + w[1] * mom_z[sym]
            + w[2] * qual_z[sym]
            + w[3] * pead_z[sym]
        )
        out.append((sym, composite, raw_by_sym[sym]))
    out.sort(key=lambda x: x[1], reverse=True)

    # Industry/correlation cap: enforce max_per_industry. We keep the
    # FULL ranked list (callers may need it for diagnostics), but the
    # top-N consumer should call ``apply_industry_cap`` to enforce it.
    if params.max_per_industry > 0:
        out = apply_industry_cap(out, params.max_per_industry)
    return out


def apply_industry_cap(
    ranked: list[tuple[str, float, dict[str, float]]],
    max_per_industry: int,
) -> list[tuple[str, float, dict[str, float]]]:
    """Drop ranked entries that would exceed ``max_per_industry`` for
    their GICS-style industry. Order is preserved — higher-scored
    members of each industry win the slot."""
    if max_per_industry <= 0:
        return ranked
    try:
        from config.universe_meta import industry_for
    except ImportError:
        return ranked
    counts: dict[str, int] = {}
    kept: list[tuple[str, float, dict[str, float]]] = []
    for tup in ranked:
        sym = tup[0]
        ind = industry_for(sym) or "_unknown"
        if counts.get(ind, 0) >= max_per_industry:
            continue
        counts[ind] = counts.get(ind, 0) + 1
        kept.append(tup)
    return kept


# ---------- backtest engine ---------- #
def run_multi_factor_swing_backtest(
    daily_by_sym: Mapping[str, pd.DataFrame],
    start: date,
    end: date,
    earnings_events: list[EarningsEvent] | None = None,
    params: MultiFactorParams = MultiFactorParams(),
) -> MFSResult:
    """Walk daily; rebalance every params.rebalance_days. Buy at OPEN of
    rebalance day; soft-stop fires intra-period at close."""
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
        return MFSResult(pd.Series(dtype=float), [], pd.Series(dtype=float))

    equity = params.initial_equity
    equity_by_day: dict[pd.Timestamp, float] = {}
    open_positions: dict[str, dict] = {}
    all_trades: list[MFSTrade] = []
    rebalance_log: list[dict] = []

    def _close_all(at_date: pd.Timestamp) -> float:
        nonlocal equity
        realized = 0.0
        for sym, pos in list(open_positions.items()):
            df = daily_by_sym.get(sym)
            if df is None or df.empty:
                continue
            df = _norm(df)
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
            comm = _commission(shares, params) * 2
            pnl_net = pnl_gross - comm
            realized += pnl_net
            all_trades.append(MFSTrade(
                symbol=sym, entry_date=pos["entry_date"],
                entry_price=pos["entry_price"], exit_date=at_date.date(),
                exit_price=exit_fill, shares=shares,
                pnl_gross=pnl_gross, commission=comm,
                composite_score=pos["composite_score"],
                technical_score=pos["scores"]["tech"],
                momentum_score=pos["scores"]["mom"],
                quality_score=pos["scores"]["qual"],
                pead_score=pos["scores"]["pead"],
            ))
        open_positions.clear()
        return realized

    def _check_soft_stops(at_date: pd.Timestamp) -> float:
        nonlocal equity
        if params.soft_stop_pct <= 0 or not open_positions:
            return 0.0
        realized = 0.0
        for sym, pos in list(open_positions.items()):
            df = daily_by_sym.get(sym)
            if df is None or df.empty:
                continue
            df = _norm(df)
            row = df.loc[df.index == at_date]
            if row.empty:
                continue
            close_px = float(row["close"].iloc[0])
            dd = (close_px - pos["entry_price"]) / pos["entry_price"]
            if dd <= -params.soft_stop_pct:
                exit_fill = _slip(close_px, side=-1, bps=params.slippage_bps)
                shares = pos["shares"]
                pnl_gross = (exit_fill - pos["entry_price"]) * shares
                comm = _commission(shares, params) * 2
                realized += pnl_gross - comm
                all_trades.append(MFSTrade(
                    symbol=sym, entry_date=pos["entry_date"],
                    entry_price=pos["entry_price"], exit_date=at_date.date(),
                    exit_price=exit_fill, shares=shares,
                    pnl_gross=pnl_gross, commission=comm,
                    composite_score=pos["composite_score"],
                    technical_score=pos["scores"]["tech"],
                    momentum_score=pos["scores"]["mom"],
                    quality_score=pos["scores"]["qual"],
                    pead_score=pos["scores"]["pead"],
                ))
                open_positions.pop(sym, None)
        return realized

    def _open_positions(picks: list[tuple[str, float, dict]],
                         at_date: pd.Timestamp) -> None:
        if not picks:
            return
        per_name = min(equity / len(picks), equity * params.max_position_pct)
        for sym, score, raws in picks:
            df = daily_by_sym.get(sym)
            if df is None or df.empty:
                continue
            df = _norm(df)
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
                "composite_score": score, "scores": raws,
            }

    last_rebal = -params.rebalance_days
    for i, today in enumerate(cal):
        equity += _check_soft_stops(today)

        if i - last_rebal >= params.rebalance_days:
            ranked = rank_universe(daily_by_sym, today.date(), earnings_events, params)
            picks = [r for r in ranked if r[1] >= params.min_composite_score][: params.top_n]
            equity += _close_all(today)
            _open_positions(picks, today)
            rebalance_log.append({
                "date": today.date().isoformat(),
                "picks": [s for s, *_ in picks],
                "n_qualified": len([r for r in ranked if r[1] >= params.min_composite_score]),
                "equity_at_rebal": equity,
            })
            last_rebal = i

        # mark-to-market
        mtm = equity
        for sym, pos in open_positions.items():
            df = daily_by_sym.get(sym)
            if df is None or df.empty:
                continue
            df = _norm(df)
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
    return MFSResult(
        equity_curve=eq_series, trades=all_trades,
        daily_returns=rets, rebalance_log=rebalance_log,
    )
