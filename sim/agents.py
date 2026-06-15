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
        self.capital   = capital
        self.agent_id  = agent_id
        self.position  = 0.0
        self.pnl       = 0.0

    @abstractmethod
    def decide(self, state: dict) -> float:
        ...

    def update_pnl(self, price_change: float):
        self.pnl += self.position * price_change

    def reset_state(self) -> None:
        self.position = 0.0
        self.pnl      = 0.0


# ---------------------------------------------------------------------------
# 1. InstitutionAgent  —  均值回歸 + trend_guard
# ---------------------------------------------------------------------------
class InstitutionAgent(BaseAgent):
    """
    均值回歸法人。

    trend_guard: 計算最近 trend_guard_window 根累積 log-return。
    若絕對值 > trend_guard_threshold，代表短期趨勢明確，不逆向。
    """

    def __init__(
        self,
        capital: float = 10.0,
        ma_window: int = 20,
        threshold: float = 0.01,
        max_order: float = 3.0,
        trend_guard_window: int = 5,
        trend_guard_threshold: float = 0.015,
        agent_id: str = "inst",
    ):
        super().__init__(capital, agent_id)
        self.ma_window             = ma_window
        self.threshold             = threshold
        self.max_order             = max_order
        self.trend_guard_window    = trend_guard_window
        self.trend_guard_threshold = trend_guard_threshold

    def decide(self, state: dict) -> float:
        closes = state["close_history"]
        if len(closes) < self.ma_window:
            return 0.0

        if self.trend_guard_window > 0 and len(closes) >= self.trend_guard_window + 1:
            recent = closes[-(self.trend_guard_window + 1):]
            cum_logret = float(np.log(recent[-1] / recent[0]))
            if abs(cum_logret) > self.trend_guard_threshold:
                return 0.0

        ma    = float(np.mean(closes[-self.ma_window:]))
        price = float(closes[-1])
        dev   = (price - ma) / ma
        if abs(dev) < self.threshold:
            return 0.0
        raw   = -dev / self.threshold
        order = float(np.clip(raw, -self.max_order, self.max_order))
        return order * self.capital


# ---------------------------------------------------------------------------
# 2. MomentumTrader  —  追漲殺跌
# ---------------------------------------------------------------------------
class MomentumTrader(BaseAgent):
    """
    動能散戶。bias 由 momentum-init 注入，透過 exponential decay 逐根遞減。
    """

    def __init__(
        self,
        capital: float = 1.0,
        lookback: int = 3,
        strength: float = 1.0,
        noise: float = 0.2,
        bias: float = 0.0,
        bias_decay: float = 0.95,
        agent_id: str = "mom",
    ):
        super().__init__(capital, agent_id)
        self.lookback   = lookback
        self.strength   = strength
        self.noise      = noise
        self.bias       = bias
        self.bias_decay = bias_decay
        self._cur_bias  = bias

    def reset_state(self) -> None:
        super().reset_state()
        self._cur_bias = self.bias

    def decide(self, state: dict) -> float:
        closes = state["close_history"]
        rng    = state["rng"]
        if len(closes) < self.lookback + 1:
            self._cur_bias *= self.bias_decay
            return self._cur_bias * self.capital

        rets   = np.diff(closes[-(self.lookback + 1):])
        signal = float(np.sum(np.sign(rets)))
        noisy  = signal + rng.normal(0, self.noise * self.lookback)
        order  = float(np.clip(noisy * self.strength, -self.lookback, self.lookback))
        total  = order + self._cur_bias
        self._cur_bias *= self.bias_decay
        return total * self.capital


# ---------------------------------------------------------------------------
# 3. RandomTrader  —  完全隨機
# ---------------------------------------------------------------------------
class RandomTrader(BaseAgent):
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


# ---------------------------------------------------------------------------
# 4. ContrarianTrader  —  逆勢 + trend_guard
# ---------------------------------------------------------------------------
class ContrarianTrader(BaseAgent):
    """
    逆勢交易者。

    trend_guard: 如果更長期的趨勢依然確立（trend_guard_window 根累積
    log-return 絕對值 > trend_guard_threshold），抑制逆勢下單。

    實際市場行為：
    - 短期振等：逆勢商會活踴（正常行為）
    - 持續趨勢：逆勢商對模擬序列貿猫，應被抑制到趨勢消退再進場
    """

    def __init__(
        self,
        capital: float = 1.5,
        window: int = 5,
        atr_mult: float = 1.5,
        max_order: float = 2.0,
        trend_guard_window: int = 15,
        trend_guard_threshold: float = 0.03,
        agent_id: str = "cont",
    ):
        super().__init__(capital, agent_id)
        self.window                = window
        self.atr_mult              = atr_mult
        self.max_order             = max_order
        self.trend_guard_window    = trend_guard_window
        self.trend_guard_threshold = trend_guard_threshold

    def decide(self, state: dict) -> float:
        closes = state["close_history"]
        atr    = state["atr"]
        if len(closes) < self.window + 1 or atr <= 0:
            return 0.0

        # trend_guard: 更長窗口趨勢明確時，不逆向下單
        if self.trend_guard_window > 0 and len(closes) >= self.trend_guard_window + 1:
            long_recent = closes[-(self.trend_guard_window + 1):]
            long_logret = float(np.log(long_recent[-1] / long_recent[0]))
            if abs(long_logret) > self.trend_guard_threshold:
                return 0.0

        cum_move  = float(closes[-1] - closes[-self.window - 1])
        threshold = self.atr_mult * atr
        if cum_move > threshold:
            strength = min(cum_move / threshold, self.max_order)
            return -strength * self.capital
        elif cum_move < -threshold:
            strength = min(-cum_move / threshold, self.max_order)
            return strength * self.capital
        return 0.0


# ---------------------------------------------------------------------------
# Agent 工廠
# ---------------------------------------------------------------------------
def build_default_agents(
    n_institution: int = 5,
    n_momentum:    int = 40,
    n_random:      int = 100,
    n_contrarian:  int = 15,
    momentum_bias: float = 0.0,
    bias_decay:    float = 0.95,
    rng: np.random.Generator | None = None,
) -> list[BaseAgent]:
    """
    建立預設 agent 群。

    InstitutionAgent  趨勢明確時靜觀（trend_guard_window=5,  threshold~1-2%）
    ContrarianTrader  趨勢持續時靜觀（trend_guard_window=15, threshold~3-5%）
    MomentumTrader    接收 momentum_bias。
    """
    if rng is None:
        rng = np.random.default_rng()

    agents: list[BaseAgent] = []

    for i in range(n_institution):
        agents.append(InstitutionAgent(
            capital=10.0,
            ma_window=int(rng.integers(15, 30)),
            threshold=float(rng.uniform(0.008, 0.02)),
            trend_guard_window=5,
            trend_guard_threshold=float(rng.uniform(0.010, 0.020)),
            agent_id=f"inst_{i}",
        ))

    for i in range(n_momentum):
        agents.append(MomentumTrader(
            capital=1.0,
            lookback=int(rng.integers(2, 6)),
            strength=float(rng.uniform(0.5, 1.5)),
            bias=momentum_bias,
            bias_decay=bias_decay,
            agent_id=f"mom_{i}",
        ))

    for i in range(n_random):
        agents.append(RandomTrader(
            capital=0.5,
            trade_prob=float(rng.uniform(0.1, 0.5)),
            agent_id=f"rand_{i}",
        ))

    for i in range(n_contrarian):
        agents.append(ContrarianTrader(
            capital=1.5,
            window=int(rng.integers(3, 8)),
            atr_mult=float(rng.uniform(1.0, 2.5)),
            trend_guard_window=15,
            trend_guard_threshold=float(rng.uniform(0.030, 0.050)),
            agent_id=f"cont_{i}",
        ))

    return agents
