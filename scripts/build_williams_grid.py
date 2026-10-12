"""Portfolio-level daily-returns grid for the Williams breakout, for CPCV.

Each variant runs a long-only portfolio sim:
  - Walk trading days. Hold up to top_n equal-weighted positions.
  - Each day: mark open positions to market (close-to-close), process
    exits (trailing/fixed stop, target, time-stop), then enter new
    breakout signals that pass the filters, up to capacity.
  - Daily portfolio return = mean of held positions' daily returns
    (uninvested capital earns 0).

Outputs reports/grid_returns_williams.parquet (n_days x n_variants), then
run_cpcv consumes it.

  python -m scripts.build_williams_grid --start 2023-01-01 --end 2025-12-31 --universe halal
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import get_settings
from scripts.screener import _load_daily_bars, _load_universe

K = 0.5


def _atr_series(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _wr_series(df, period=10):
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return (hh - df["close"]) / (hh - ll).replace(0, np.nan) * -100.0


@dataclass
class Cfg:
    stop_atr: float = 2.0
    tp_mult: float = 3.0
    vol_mult: float = 1.5
    pr_lo: float = -60.0
    pr_hi: float = -15.0
    trend_buffer: float = 0.03
    trail_atr: float = 0.0   # 0 = fixed TP
    hold_bars: int = 10
    top_n: int = 5


@dataclass
class _Pos:
    entry: float
    stop: float
    tp: float
    bars_held: int
    trail_atr_abs: float  # absolute trail distance (atr * trail_atr), 0 = fixed
    prev_close: float


def _precompute(daily):
    feats = {}
    for sym, df in daily.items():
        d = df.copy()
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        d = d.sort_index()
        if len(d) < 220:
            continue
        d["sma50"] = d["close"].rolling(50).mean()
        d["sma200"] = d["close"].rolling(200).mean()
        d["atr"] = _atr_series(d)
        d["vol20"] = d["volume"].rolling(20).mean()
        d["wr"] = _wr_series(d)
        feats[sym] = d
    return feats


def portfolio_returns(feats, start, end, cfg: Cfg) -> pd.Series:
    # master date index = union of all symbol dates in window
    all_dates = sorted({ts for d in feats.values() for ts in d.index
                        if start <= ts <= end})
    positions: dict[str, _Pos] = {}
    daily_ret: dict[pd.Timestamp, float] = {}

    for ts in all_dates:
        day_rets: list[float] = []
        # 1) mark + manage open positions
        for sym in list(positions.keys()):
            d = feats[sym]
            if ts not in d.index:
                continue
            row = d.loc[ts]
            pos = positions[sym]
            lo, hi, cl = float(row["low"]), float(row["high"]), float(row["close"])
            # daily MTM contribution (close-to-close)
            if pos.prev_close > 0:
                day_rets.append(cl / pos.prev_close - 1.0)
            # exit logic (stop first, then target, then time)
            exit_now = False
            active_stop = pos.stop
            if active_stop and lo <= active_stop:
                exit_now = True
            elif pos.trail_atr_abs == 0 and hi >= pos.tp:
                exit_now = True
            pos.bars_held += 1
            if pos.bars_held >= cfg.hold_bars:
                exit_now = True
            # update trailing stop
            if pos.trail_atr_abs > 0:
                pos.stop = max(pos.stop, cl - pos.trail_atr_abs)
            pos.prev_close = cl
            if exit_now:
                del positions[sym]

        # 2) enter new signals up to capacity
        if len(positions) < cfg.top_n:
            candidates = []
            for sym, d in feats.items():
                if sym in positions or ts not in d.index:
                    continue
                loc = d.index.get_loc(ts)
                if loc < 200:
                    continue
                prior = d.iloc[loc - 1]
                row = d.iloc[loc]
                pc = float(prior["close"])
                c50, c200 = prior["sma50"], prior["sma200"]
                if not (np.isfinite(c50) and np.isfinite(c200)
                        and pc > c50 * (1 + cfg.trend_buffer) and c50 > c200):
                    continue
                prng = float(prior["high"]) - float(prior["low"])
                if prng <= 0:
                    continue
                pr = prior["wr"]
                if np.isfinite(pr) and not (cfg.pr_lo <= pr <= cfg.pr_hi):
                    continue
                trigger = pc + K * prng
                if float(row["high"]) < trigger:
                    continue
                if cfg.vol_mult > 0:
                    v20 = row["vol20"]
                    if not (np.isfinite(v20) and float(row["volume"]) >= cfg.vol_mult * v20):
                        continue
                a = float(prior["atr"]) if np.isfinite(prior["atr"]) else prng
                rng_atr = prng / a if a > 0 else 0
                candidates.append((rng_atr, sym, trigger, a, prng, float(row["close"])))
            # rank by range/ATR (most decisive breakouts first)
            candidates.sort(reverse=True)
            for _, sym, trigger, a, prng, cl in candidates:
                if len(positions) >= cfg.top_n:
                    break
                positions[sym] = _Pos(
                    entry=trigger,
                    stop=trigger - cfg.stop_atr * a,
                    tp=trigger + cfg.tp_mult * prng,
                    bars_held=0,
                    trail_atr_abs=(cfg.trail_atr * a) if cfg.trail_atr > 0 else 0.0,
                    prev_close=cl,
                )

        daily_ret[ts] = float(np.mean(day_rets)) if day_rets else 0.0

    return pd.Series(daily_ret).sort_index()


def main(argv=None) -> int:
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--universe", default="halal")
    p.add_argument("--out", type=Path,
                   default=Path("reports/grid_returns_williams.parquet"))
    args = p.parse_args(argv)

    start = pd.Timestamp(datetime.fromisoformat(args.start).date())
    end = pd.Timestamp(datetime.fromisoformat(args.end).date())
    daily = _load_daily_bars(_load_universe(args.universe), s.data_dir)
    feats = _precompute(daily)
    print(f"precomputed {len(feats)} symbols")

    # Grid: vary the knobs we tuned, so CPCV penalises the multiple testing.
    vol_mults = (1.3, 1.5, 2.0)
    trail_atrs = (0.0, 2.5, 3.5)
    trend_bufs = (0.0, 0.03)
    series = {}
    for vm, tr, tb in product(vol_mults, trail_atrs, trend_bufs):
        cfg = Cfg(vol_mult=vm, trail_atr=tr, trend_buffer=tb)
        name = f"vol{int(vm*10):02d}_trail{int(tr*10):02d}_tb{int(tb*100):02d}"
        ser = portfolio_returns(feats, start, end, cfg)
        series[name] = ser
        print(f"  {name}: {len(ser)} days, "
              f"ann.ret~{(ser.mean()*252)*100:.1f}%")
    df = pd.DataFrame(series).fillna(0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"wrote {args.out}  shape={df.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
