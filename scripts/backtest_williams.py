"""Lightweight event-driven backtest for the Williams volatility breakout.

For each day in the window, for each symbol:
  1. Compute the breakout trigger from the PRIOR bar (no look-ahead).
  2. Require an uptrend (close > 50-SMA > 200-SMA) on the prior bar.
  3. If today's HIGH >= trigger -> enter long at the trigger price.
  4. Walk forward up to `hold_bars`; exit at whichever comes first:
       - stop  (midpoint of prior-low/entry, ATR-floored)
       - TP2    (entry + 3 x range)
       - TP1    (entry + 2 x range)  -> we model a full exit at TP1 for
                 simplicity (conservative on the upside)
       - time-stop at hold_bars (exit at close)
  5. Record return, win/loss.

Honest, simple, vectorisable-enough. Reports win rate, avg win/loss,
expectancy, profit factor.

  python -m scripts.backtest_williams --start 2023-01-01 --end 2025-12-31 --universe halal
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import get_settings
from scripts.screener import _load_daily_bars, _load_universe


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ATR as a rolling series (one value per bar)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

K = 0.5
TP1_MULT, TP2_MULT = 2.0, 3.0
MIN_STOP_ATR_FRAC = 2.0
HOLD_BARS = 10


def _williams_r_series(df: pd.DataFrame, period: int = 10) -> pd.Series:
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return (hh - df["close"]) / (hh - ll).replace(0, np.nan) * -100.0


def _backtest_symbol(df: pd.DataFrame, start: pd.Timestamp,
                     end: pd.Timestamp, *, k: float = K,
                     stop_atr: float = MIN_STOP_ATR_FRAC,
                     tp_mult: float = TP1_MULT,
                     hold_bars: int = HOLD_BARS,
                     vol_mult: float = 0.0,          # 0 = no volume filter
                     pr_lo: float = -100.0,          # %R lower bound
                     pr_hi: float = 0.0,             # %R upper bound
                     trend_buffer: float = 0.0,      # close > sma50*(1+buf)
                     trail_atr: float = 0.0,         # 0 = fixed TP; else trail
                     ) -> list[dict]:
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    if len(df) < 220:
        return []
    closes = df["close"]
    sma50 = closes.rolling(50).mean()
    sma200 = closes.rolling(200).mean()
    atr14 = _atr_series(df, 14)
    vol20 = df["volume"].rolling(20).mean()
    pctr = _williams_r_series(df, 10)

    trades: list[dict] = []
    idx = df.index
    for i in range(200, len(df) - 1):
        ts = idx[i]
        if ts < start or ts > end:
            continue
        prior = df.iloc[i - 1]
        c50, c200 = sma50.iloc[i - 1], sma200.iloc[i - 1]
        pc = float(prior["close"])
        # Trend filter (with optional strength buffer above 50-SMA)
        if not (np.isfinite(c50) and np.isfinite(c200)
                and pc > c50 * (1.0 + trend_buffer) and c50 > c200):
            continue
        prng = float(prior["high"]) - float(prior["low"])
        if prng <= 0:
            continue
        # %R momentum band filter (prior bar)
        pr = pctr.iloc[i - 1]
        if np.isfinite(pr) and not (pr_lo <= pr <= pr_hi):
            continue
        trigger = pc + k * prng
        today = df.iloc[i]
        if float(today["high"]) < trigger:
            continue  # not triggered
        # Volume confirmation on the breakout bar
        if vol_mult > 0:
            v20 = vol20.iloc[i]
            if not (np.isfinite(v20) and float(today["volume"]) >= vol_mult * v20):
                continue
        a = float(atr14.iloc[i - 1]) if np.isfinite(atr14.iloc[i - 1]) else prng
        entry = trigger
        stop = entry - stop_atr * a
        tp1 = entry + tp_mult * prng
        # walk forward
        exit_price = None
        trail = stop  # trailing stop seed
        for j in range(i, min(i + hold_bars + 1, len(df))):
            bar = df.iloc[j]
            lo, hi, cl = float(bar["low"]), float(bar["high"]), float(bar["close"])
            # stop (fixed or trailing) checked first (conservative)
            active_stop = trail if trail_atr > 0 else stop
            if lo <= active_stop:
                exit_price = active_stop; break
            if trail_atr <= 0 and hi >= tp1:
                exit_price = tp1; break
            # update trailing stop on the close
            if trail_atr > 0:
                trail = max(trail, cl - trail_atr * a)
        if exit_price is None:
            exit_price = float(df.iloc[min(i + hold_bars, len(df) - 1)]["close"])
        ret = (exit_price / entry - 1.0) * 100.0
        trades.append({"ts": ts, "entry": entry, "exit": exit_price,
                       "ret_pct": ret, "win": ret > 0})
    return trades


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--universe", default="halal")
    args = p.parse_args(argv)

    start = pd.Timestamp(datetime.fromisoformat(args.start).date())
    end = pd.Timestamp(datetime.fromisoformat(args.end).date())
    symbols = _load_universe(args.universe)
    daily = _load_daily_bars(symbols, s.data_dir)

    # Each variant: (label, kwargs). Start from the validated baseline
    # (stop 2.0 ATR, tp 3x range) and layer filters to find what lifts edge.
    base = dict(stop_atr=2.0, tp_mult=3.0)
    variants = [
        ("baseline (2.0/3x)",        {}),
        ("+ %R band -60..-15",       dict(pr_lo=-60, pr_hi=-15)),
        ("+ volume >1.5x",           dict(vol_mult=1.5)),
        ("+ trend buf 3%",           dict(trend_buffer=0.03)),
        ("+ %R + vol",               dict(pr_lo=-60, pr_hi=-15, vol_mult=1.5)),
        ("+ %R + vol + trend",       dict(pr_lo=-60, pr_hi=-15, vol_mult=1.5,
                                          trend_buffer=0.03)),
        ("ALL + trailing 2.5ATR",    dict(pr_lo=-60, pr_hi=-15, vol_mult=1.5,
                                          trend_buffer=0.03, trail_atr=2.5)),
        ("ALL + trailing 3.5ATR",    dict(pr_lo=-60, pr_hi=-15, vol_mult=1.5,
                                          trend_buffer=0.03, trail_atr=3.5)),
    ]
    print()
    print(f"=== WILLIAMS BREAKOUT FILTER SWEEP ({args.universe}, "
          f"{args.start}..{args.end}) ===")
    print(f"{'variant':<26} {'trades':>7} {'win%':>6} {'avgW':>6} "
          f"{'avgL':>6} {'exp%':>7} {'PF':>5}")
    for label, extra in variants:
        kw = {**base, **extra}
        trades: list[dict] = []
        for sym, df in daily.items():
            trades.extend(_backtest_symbol(df, start, end, **kw))
        if not trades:
            print(f"{label:<26} {'0':>7}  (no trades)")
            continue
        rets = np.array([t["ret_pct"] for t in trades])
        wins = rets[rets > 0]; losses = rets[rets <= 0]
        wr = len(wins) / len(rets)
        aw = wins.mean() if len(wins) else 0.0
        al = losses.mean() if len(losses) else 0.0
        exp = rets.mean()
        gl = abs(losses.sum())
        pf = wins.sum() / gl if gl > 0 else float("inf")
        print(f"{label:<26} {len(rets):>7} {wr*100:>5.1f}% "
              f"{aw:>+5.2f} {al:>+5.2f} {exp:>+6.3f} {pf:>5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
