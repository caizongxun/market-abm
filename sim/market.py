"""
market.py
=========
Market matching engine: collects all agent orders,
computes net position, and generates the next OHLCV bar
via a market impact model.

Impact model
------------
  net_order  = sum(all agent orders)
  impact     = net_order / total_capital * impact_coeff + drift_per_bar
  next_open  = last_close * exp(impact)
  intra-bar  = ATR-based t-dist noise, clipped to [-3, 3] sigma  (fat tails, no explosion)
  close      = open + intra_bar_noise  (floored at last_close * 0.5)
  high/low   = body +/- t-distributed wick
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int = 14) -> float:
    """Average True Range."""
    if len(closes) < 2:
        return float(np.std(closes)) if len(closes) > 0 else 0.01
    tr = np.maximum(
        highs[1:]  - lows[1:],
        np.maximum(
            np.abs(highs[1:]  - closes[:-1]),
            np.abs(lows[1:]   - closes[:-1]),
        ),
    )
    return float(np.mean(tr[-window:])) if len(tr) >= window else float(np.mean(tr))


def total_capital(agents) -> float:
    return sum(abs(a.capital) for a in agents)


# Normalisation constant: std of t(df=4) = sqrt(df/(df-2)) = sqrt(2)
_T4_STD = float(np.sqrt(2.0))
# Hard clip for t-draws: preserves fat tails up to 3-sigma, prevents runaway
_T_CLIP = 3.0


class MarketEngine:
    """
    Advances the market by one bar per step() call.

    Parameters
    ----------
    impact_coeff : float
        Sensitivity of price to net order flow.  Range: 0.0001 ~ 0.005
    intra_noise_scale : float
        ATR multiplier for intra-bar volatility.
    drift_per_bar : float
        Per-bar log-return drift estimated from warmup history.
    volume_base : float
        Base volume for simulated bars.
    initial_price : float | None
        若指定，則 MarketEngine 以此價格作為第一根 bar 的 last_close，
        覆蓋 close_history[-1]。用於 rolling window 接縫對齊。
    """

    def __init__(
        self,
        agents: list,
        impact_coeff: float = 0.001,
        intra_noise_scale: float = 1.0,
        drift_per_bar: float = 0.0,
        volume_base: float = 1e6,
        seed: int | None = None,
        initial_price: float | None = None,
    ):
        self.agents            = agents
        self.impact_coeff      = impact_coeff
        self.intra_noise_scale = intra_noise_scale
        self.drift_per_bar     = drift_per_bar
        self.volume_base       = volume_base
        self.rng               = np.random.default_rng(seed)
        self._total_cap        = total_capital(agents)
        self._initial_price    = initial_price   # 接縫對齊用，只在第一根 bar 生效
        self._first_step_done  = False

    def _t_draw(self) -> float:
        """Draw from t(df=4), clip to [-_T_CLIP, _T_CLIP], normalise to unit std."""
        raw = float(self.rng.standard_t(df=4))
        return np.clip(raw, -_T_CLIP, _T_CLIP) / _T4_STD

    def step(
        self,
        close_history:  np.ndarray,
        high_history:   np.ndarray,
        low_history:    np.ndarray,
        volume_history: np.ndarray,
        bar_idx:        int,
    ) -> dict:
        """
        Let all agents decide, match orders, produce next bar.

        Returns
        -------
        dict with keys: open, high, low, close, volume, net_order, agent_orders
        """
        atr = compute_atr(high_history, low_history, close_history)

        state = {
            "close_history":  close_history,
            "volume_history": volume_history,
            "bar_idx":        bar_idx,
            "atr":            atr,
            "rng":            self.rng,
        }

        # Collect orders
        orders    = np.array([a.decide(state) for a in self.agents], dtype=float)
        net_order = float(np.sum(orders))

        # Market impact + drift -> next open
        order_impact = (net_order / max(self._total_cap, 1e-6)) * self.impact_coeff
        total_impact = order_impact + self.drift_per_bar

        # 接縫對齊：第一根 bar 使用 initial_price 覆蓋 close_history[-1]
        if self._initial_price is not None and not self._first_step_done:
            last_close = self._initial_price
            self._first_step_done = True
        else:
            last_close = float(close_history[-1])

        new_open     = last_close * np.exp(total_impact)

        # Intra-bar noise: clipped t(df=4), normalised to unit std
        intra_vol  = atr * self.intra_noise_scale
        intra_move = self._t_draw() * intra_vol
        new_close  = new_open + intra_move
        # Guard: close must stay above 50% of last close (prevents negative / runaway)
        new_close  = max(new_close, last_close * 0.5)

        # High / Low: body + clipped-t wick
        body_high  = max(new_open, new_close)
        body_low   = min(new_open, new_close)
        wick_scale = atr * 0.3
        upper_wick = abs(self._t_draw() * wick_scale)
        lower_wick = abs(self._t_draw() * wick_scale)
        new_high   = body_high + upper_wick
        new_low    = body_low  - lower_wick
        # Low can't go negative
        new_low    = max(new_low, last_close * 0.01)

        # Volume: proportional to |net_order|
        vol_factor = 1.0 + abs(net_order) / max(self._total_cap, 1e-6) * 5
        new_volume = self.volume_base * vol_factor * abs(self.rng.normal(1.0, 0.2))

        # Update agent PnL
        price_change = new_close - last_close
        for a in self.agents:
            a.update_pnl(price_change)

        return {
            "open":         round(float(new_open),   4),
            "high":         round(float(new_high),   4),
            "low":          round(float(new_low),    4),
            "close":        round(float(new_close),  4),
            "volume":       round(float(new_volume), 0),
            "net_order":    round(net_order,          4),
            "agent_orders": orders,
        }
