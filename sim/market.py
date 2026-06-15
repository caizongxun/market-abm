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
  intra-bar  = ATR-based t-distributed noise  (df=4, fat tails)
  close      = open + intra_bar_noise
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


class MarketEngine:
    """
    Advances the market by one bar per step() call.

    Parameters
    ----------
    impact_coeff : float
        Sensitivity of price to net order flow.
        Reasonable range: 0.0001 ~ 0.005
    intra_noise_scale : float
        ATR multiplier for intra-bar volatility (controls High/Low spread).
    drift_per_bar : float
        Per-bar log-return drift estimated from warmup history.
        Injected by simulation.py so the engine tracks the underlying trend.
    volume_base : float
        Base volume for simulated bars.
    """

    def __init__(
        self,
        agents: list,
        impact_coeff: float = 0.0005,
        intra_noise_scale: float = 1.0,
        drift_per_bar: float = 0.0,
        volume_base: float = 1e6,
        seed: int | None = None,
    ):
        self.agents            = agents
        self.impact_coeff      = impact_coeff
        self.intra_noise_scale = intra_noise_scale
        self.drift_per_bar     = drift_per_bar
        self.volume_base       = volume_base
        self.rng               = np.random.default_rng(seed)
        self._total_cap        = total_capital(agents)

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
        orders = np.array([a.decide(state) for a in self.agents], dtype=float)
        net_order = float(np.sum(orders))

        # Market impact + drift -> next open
        order_impact = (net_order / max(self._total_cap, 1e-6)) * self.impact_coeff
        total_impact = order_impact + self.drift_per_bar
        last_close   = float(close_history[-1])
        new_open     = last_close * np.exp(total_impact)

        # Intra-bar noise: t-distribution (df=4) for fat tails
        intra_vol   = atr * self.intra_noise_scale
        t_draw      = float(self.rng.standard_t(df=4))
        # Scale t-draw so its std matches intra_vol
        # std of t(df=4) = sqrt(df/(df-2)) = sqrt(2)
        intra_move  = t_draw / np.sqrt(2.0) * intra_vol
        new_close   = new_open + intra_move

        # High / Low: body + t-distributed wick
        body_high = max(new_open, new_close)
        body_low  = min(new_open, new_close)
        wick_scale = atr * 0.3
        upper_wick = abs(float(self.rng.standard_t(df=4)) / np.sqrt(2.0) * wick_scale)
        lower_wick = abs(float(self.rng.standard_t(df=4)) / np.sqrt(2.0) * wick_scale)
        new_high   = body_high + upper_wick
        new_low    = body_low  - lower_wick

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
