"""
agents.py
=========
4 種交易 agent，每根 K 棒結束後根據市場資訊決定買賣量。

回傳值慣例
----------
  +N : 買入 N 單位（做多）
  -N : 賣出 N 單位（做空 / 平多）
   0 : 觀望

所有 agent 繼承 BaseAgent，實作 decide(market_state) -> float

market_state dict 格式
-----------------------
  close_history : np.ndarray   # 過去所有收盤價（含本根）
  volume_history: np.ndarray   # 過去所有成交量
  bar_idx       : int          # 目前是第幾根（0-based）
  atr           : float        # 最近 14 根 ATR
  rng           : np.random.Generator
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, capital: float, agent_id: str):
        self.capital   = capital   # 資金規模，影響下單量
        self.agent_id  = agent_id
        self.position  = 0.0      # 目前淨部位（+多 / -空）
        self.pnl       = 0.0

    @abstractmethod
    def decide(self, state: dict) -> float:
        """回傳本根 K 棒的下單量（正=買, 負=賣）。"""
        ...

    def update_pnl(self, price_change: float):
        self.pnl += self.position * price_change


# ─────────────────────────────────────────────────────────────────────────────
# 1. InstitutionAgent  —  均值回歸
# ─────────────────────────────────────────────────────────────────────────────
class InstitutionAgent(BaseAgent):
    """
    機構大戶：當收盤價偏離 MA 超過門檻，反向下注。
    偏離越大，下單量越大（線性比例）。
    """

    def __init__(
        self,
        capital: float = 10.0,
        ma_window: int = 20,
        threshold: float = 0.01,   # 偏離 MA 1% 才啟動
        max_order: float = 3.0,
        agent_id: str = "inst",
    ):
        super().__init__(capital, agent_id)
        self.ma_window  = ma_window
        self.threshold  = threshold
        self.max_order  = max_order

    def decide(self, state: dict) -> float:
        closes = state["close_history"]
        if len(closes) < self.ma_window:
            return 0.0

        ma    = float(np.mean(closes[-self.ma_window:]))
        price = float(closes[-1])
        dev   = (price - ma) / ma   # 正 = 高於 MA

        if abs(dev) < self.threshold:
            return 0.0

        # 偏離越大下單越多（上限 max_order）
        raw = -dev / self.threshold   # 高於 MA → 空；低於 MA → 多
        order = float(np.clip(raw, -self.max_order, self.max_order))
        return order * self.capital


# ─────────────────────────────────────────────────────────────────────────────
# 2. MomentumTrader  —  追漲殺跌
# ─────────────────────────────────────────────────────────────────────────────
class MomentumTrader(BaseAgent):
    """
    動能散戶：連續 N 根上漲就跟買，連續 N 根下跌就跟空。
    strength 控制追漲強度（越高 = 訊號越弱時也跟進）。
    """

    def __init__(
        self,
        capital: float = 1.0,
        lookback: int = 3,
        strength: float = 1.0,
        noise: float = 0.2,
        agent_id: str = "mom",
    ):
        super().__init__(capital, agent_id)
        self.lookback = lookback
        self.strength = strength
        self.noise    = noise

    def decide(self, state: dict) -> float:
        closes = state["close_history"]
        rng    = state["rng"]
        if len(closes) < self.lookback + 1:
            return 0.0

        rets   = np.diff(closes[-(self.lookback + 1):])
        signal = float(np.sum(np.sign(rets)))   # 範圍 [-lookback, +lookback]
        # 加噪音（並非所有動能散戶都同步）
        noisy  = signal + rng.normal(0, self.noise * self.lookback)
        order  = float(np.clip(noisy * self.strength, -self.lookback, self.lookback))
        return order * self.capital


# ─────────────────────────────────────────────────────────────────────────────
# 3. RandomTrader  —  完全隨機
# ─────────────────────────────────────────────────────────────────────────────
class RandomTrader(BaseAgent):
    """
    隨機散戶：每根 K 棒以機率 trade_prob 入場，方向完全隨機。
    模擬市場背景噪音。
    """

    def __init__(
        self,
        capital: float = 0.5,
        trade_prob: float = 0.3,
        max_order: float = 1.0,
        agent_id: str = "rand",
    ):
        super().__init__(capital, agent_id)
        self.trade_prob = trade_prob
        self.max_order  = max_order

    def decide(self, state: dict) -> float:
        rng = state["rng"]
        if rng.random() > self.trade_prob:
            return 0.0
        direction = rng.choice([-1.0, 1.0])
        size      = rng.uniform(0.1, self.max_order)
        return direction * size * self.capital


# ─────────────────────────────────────────────────────────────────────────────
# 4. ContrarianTrader  —  逆勢
# ─────────────────────────────────────────────────────────────────────────────
class ContrarianTrader(BaseAgent):
    """
    逆勢者：計算過去 window 根的累積漲跌幅，
    漲太多就空、跌太多就多。門檻以 ATR 倍數衡量。
    """

    def __init__(
        self,
        capital: float = 1.5,
        window: int = 5,
        atr_mult: float = 1.5,   # 累積漲跌 > atr_mult * ATR 才出手
        max_order: float = 2.0,
        agent_id: str = "cont",
    ):
        super().__init__(capital, agent_id)
        self.window   = window
        self.atr_mult = atr_mult
        self.max_order = max_order

    def decide(self, state: dict) -> float:
        closes = state["close_history"]
        atr    = state["atr"]
        if len(closes) < self.window + 1 or atr <= 0:
            return 0.0

        cum_move = float(closes[-1] - closes[-self.window - 1])
        threshold = self.atr_mult * atr

        if cum_move > threshold:
            # 漲太多 → 空
            strength = min(cum_move / threshold, self.max_order)
            return -strength * self.capital
        elif cum_move < -threshold:
            # 跌太多 → 多
            strength = min(-cum_move / threshold, self.max_order)
            return strength * self.capital
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Agent 工廠
# ─────────────────────────────────────────────────────────────────────────────
def build_default_agents(
    n_institution: int = 5,
    n_momentum:    int = 40,
    n_random:      int = 100,
    n_contrarian:  int = 15,
    rng: np.random.Generator | None = None,
) -> list[BaseAgent]:
    """建立預設 agent 群，參數可個別微調。"""
    if rng is None:
        rng = np.random.default_rng()

    agents: list[BaseAgent] = []

    # 機構：少量但資金大，MA window 略有差異
    for i in range(n_institution):
        agents.append(InstitutionAgent(
            capital=10.0,
            ma_window=int(rng.integers(15, 30)),
            threshold=float(rng.uniform(0.008, 0.02)),
            agent_id=f"inst_{i}",
        ))

    # 動能散戶：lookback 各不相同
    for i in range(n_momentum):
        agents.append(MomentumTrader(
            capital=1.0,
            lookback=int(rng.integers(2, 6)),
            strength=float(rng.uniform(0.5, 1.5)),
            agent_id=f"mom_{i}",
        ))

    # 隨機散戶
    for i in range(n_random):
        agents.append(RandomTrader(
            capital=0.5,
            trade_prob=float(rng.uniform(0.1, 0.5)),
            agent_id=f"rand_{i}",
        ))

    # 逆勢者
    for i in range(n_contrarian):
        agents.append(ContrarianTrader(
            capital=1.5,
            window=int(rng.integers(3, 8)),
            atr_mult=float(rng.uniform(1.0, 2.5)),
            agent_id=f"cont_{i}",
        ))

    return agents
