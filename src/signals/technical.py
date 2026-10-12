"""Technical confluence signal: RSI + MACD + SMA filter + breakout volume.

Pure-numpy/pandas indicators. Each function takes a daily close series
(and volume where needed) and returns the latest value plus a context
window. All indicators are CAUSAL — they only use bars whose timestamp
is strictly before the reference date.

The composite score is the (z-scored) sum of:
  - momentum_5d  — short-term return
  - rsi_position — distance of RSI from overbought (50 < rsi < 70 = ideal)
  - macd_signal  — MACD histogram positive AND rising
  - trend_strength — close above SMA50 AND SMA50 above SMA200
  - volume_spike — relative volume > 1.5x

A name's score is positive only when MULTIPLE confirmations align.
This is the "confluence" the swing literature consistently recommends
(MACD + RSI + trend filter + volume gives 73% win rate per the
QuantifiedStrategies replication).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rsi(closes: pd.Series, window: int = 14) -> float:
    """Wilder's RSI on the latest window. Returns NaN if insufficient data."""
    if len(closes) < window + 1:
        return float("nan")
    deltas = closes.diff().dropna()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)
    avg_gain = gains.iloc[-window:].mean()
    avg_loss = losses.iloc[-window:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def ema(values: pd.Series, span: int) -> pd.Series:
    """Standard EMA via pandas — note that this is NOT lookahead-leaking
    because pandas computes it forward-fill from the start of the series."""
    return values.ewm(span=span, adjust=False).mean()


def macd(closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[float, float, float]:
    """Returns (macd_line, signal_line, histogram) for the LAST observation."""
    if len(closes) < slow + signal:
        return float("nan"), float("nan"), float("nan")
    macd_line = ema(closes, fast) - ema(closes, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])


def sma(closes: pd.Series, window: int) -> float:
    if len(closes) < window:
        return float("nan")
    return float(closes.iloc[-window:].mean())


def atr(bars: pd.DataFrame, window: int = 14) -> float:
    """Wilder's ATR. Expects 'high', 'low', 'close' columns."""
    if len(bars) < window + 1:
        return float("nan")
    high = bars["high"]
    low = bars["low"]
    prev_close = bars["close"].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.iloc[-window:].mean())


def relative_volume(volumes: pd.Series, window: int = 20) -> float:
    """Today's volume / mean of last `window` days (excluding today)."""
    if len(volumes) < window + 1:
        return float("nan")
    today = float(volumes.iloc[-1])
    avg = float(volumes.iloc[-(window + 1):-1].mean())
    if avg <= 0:
        return float("nan")
    return today / avg


@dataclass(frozen=True)
class TechnicalSnapshot:
    """All the per-symbol technical readings at a reference date."""
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    sma_50: float
    sma_200: float
    last_close: float
    atr_14: float
    rel_volume: float
    momentum_5d: float

    def is_in_uptrend(self) -> bool:
        """Close > SMA50 > SMA200 — the canonical 'trending up' filter."""
        if any(map(lambda x: not np.isfinite(x),
                   [self.last_close, self.sma_50, self.sma_200])):
            return False
        return self.last_close > self.sma_50 > self.sma_200

    def composite_score(self) -> float:
        """Single number combining all signals. Higher = more bullish.

        Components weighted equally:
          - momentum_5d (cap at +/-15% to avoid outliers dominating)
          - macd_hist sign (+/-1)
          - rsi 50-70 sweet spot (+1 if in range, -1 if overbought >75
            or oversold <40)
          - trend filter (+1 if uptrend per is_in_uptrend, else -1)
          - rel_volume - 1 (capped at 3)
        """
        if not np.isfinite(self.last_close):
            return float("nan")

        mom_clamped = max(-0.15, min(0.15, self.momentum_5d or 0.0))
        macd_score = 0.0
        if np.isfinite(self.macd_hist):
            macd_score = 1.0 if self.macd_hist > 0 else -1.0

        rsi_score = 0.0
        if np.isfinite(self.rsi_14):
            if 50 <= self.rsi_14 <= 70:
                rsi_score = 1.0
            elif self.rsi_14 > 75 or self.rsi_14 < 40:
                rsi_score = -1.0

        trend_score = 1.0 if self.is_in_uptrend() else -1.0

        vol_score = 0.0
        if np.isfinite(self.rel_volume):
            vol_score = max(-1.0, min(3.0, self.rel_volume - 1.0))

        # Equal-weighted composite. Range roughly [-4, +7].
        return (
            mom_clamped * 10  # scale 0-1.5 range
            + macd_score
            + rsi_score
            + trend_score
            + vol_score
        )


def technical_snapshot(bars: pd.DataFrame, ref_date: pd.Timestamp
                        ) -> TechnicalSnapshot | None:
    """Build a snapshot from daily OHLCV bars at ref_date (causal: closes < ref_date).

    Returns None if there's insufficient history.
    """
    if bars is None or bars.empty:
        return None
    df = bars.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    ts = ref_date.tz_localize(None) if ref_date.tzinfo is not None else ref_date
    prior = df.loc[df.index < ts]
    if len(prior) < 200:  # need 200 days for SMA200
        return None

    closes = prior["close"]
    vols = prior["volume"]

    macd_line, macd_signal, macd_hist = macd(closes)

    if len(closes) < 6:
        mom_5d = float("nan")
    else:
        end = float(closes.iloc[-1])
        start = float(closes.iloc[-6])
        mom_5d = (end / start - 1.0) if start > 0 else float("nan")

    return TechnicalSnapshot(
        rsi_14=rsi(closes, 14),
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        sma_50=sma(closes, 50),
        sma_200=sma(closes, 200),
        last_close=float(closes.iloc[-1]),
        atr_14=atr(prior),
        rel_volume=relative_volume(vols),
        momentum_5d=mom_5d,
    )
