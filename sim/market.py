"""
market.py
=========
市場撮合引擎：收集所有 agent 的下單量，
計算淨部位，透過市場衝擊模型產生下一根 K 棒的 OHLCV。

市場衝擊模型
------------
  net_order = sum(all agent orders)
  impact    = net_order / total_capital * impact_coeff
  next_open = last_close * exp(impact)
  K 棒內波動 = ATR-based 隨機擾動
  close     = open + intra_bar_drift + noise
  high      = max(open, close) + upper_wick
  low       = min(open, close) - lower_wick
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int = 14) -> float:
    """計算 ATR（True Range 平均）。"""
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
    每呼叫一次 step()，推進一根 K 棒。

    Parameters
    ----------
    impact_coeff : float
        淨部位對價格的衝擊強度。越大 = 市場對訂單越敏感。
        合理範圍：0.0001 ~ 0.005
    intra_noise_scale : float
        K 棒內隨機波動的 ATR 倍數（控制 High/Low 寬度）。
    volume_base : float
        基礎成交量（模擬用，實際研究可替換成真實量）。
    """

    def __init__(
        self,
        agents: list,
        impact_coeff: float = 0.0005,
        intra_noise_scale: float = 0.6,
        volume_base: float = 1e6,
        seed: int | None = None,
    ):
        self.agents            = agents
        self.impact_coeff      = impact_coeff
        self.intra_noise_scale = intra_noise_scale
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
        根據歷史資料讓所有 agent 決策，撮合後產生新一根 K 棒。

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

        # 收集所有訂單
        orders = np.array([a.decide(state) for a in self.agents], dtype=float)
        net_order = float(np.sum(orders))

        # 市場衝擊 → 影響下一根 open 的漂移
        impact = (net_order / max(self._total_cap, 1e-6)) * self.impact_coeff
        last_close = float(close_history[-1])

        new_open = last_close * np.exp(impact)

        # K 棒內波動（基於 ATR）
        intra_vol  = atr * self.intra_noise_scale
        intra_move = self.rng.normal(0.0, intra_vol)
        new_close  = new_open + intra_move

        # High / Low：在 open/close 範圍外再加 wick
        body_high = max(new_open, new_close)
        body_low  = min(new_open, new_close)
        upper_wick = abs(self.rng.normal(0, atr * 0.3))
        lower_wick = abs(self.rng.normal(0, atr * 0.3))
        new_high = body_high + upper_wick
        new_low  = body_low  - lower_wick

        # 成交量：與 |net_order| 正相關
        vol_factor  = 1.0 + abs(net_order) / max(self._total_cap, 1e-6) * 5
        new_volume  = self.volume_base * vol_factor * abs(self.rng.normal(1.0, 0.2))

        # 更新 agent PnL
        price_change = new_close - last_close
        for a in self.agents:
            a.update_pnl(price_change)

        return {
            "open":        round(float(new_open),   4),
            "high":        round(float(new_high),   4),
            "low":         round(float(new_low),    4),
            "close":       round(float(new_close),  4),
            "volume":      round(float(new_volume), 0),
            "net_order":   round(net_order,          4),
            "agent_orders": orders,
        }
