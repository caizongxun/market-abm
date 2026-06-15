"""
simulation.py
=============
主迴圈：用真實歷史 K 棒初始化 agent 狀態，然後模擬 N 根 K 棒。

Drift / Momentum-init 策略
--------------------------
不自動從 warmup 估算 drift（會帶入歷史偏差）。
改由呼叫方傳入 `drift_per_bar`：
  - 若為 0.0（預設）→ 純 agent 決策
  - 若由 estimate_momentum_drift() 計算 → 近期動能初始偏移（exponential decay）
  - 若手動指定 → 使用者自己負責
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .agents import BaseAgent, build_default_agents
from .market import MarketEngine


def estimate_momentum_drift(
    close_history: np.ndarray,
    window: int = 20,
    scale: float = 1.0,
) -> float:
    """
    從最近 window 根收盤價估算動能 drift。

    計算方式：window 根的平均每根 log-return，乘上 scale。
    正值代表近期上漲動能；負值代表近期下跌動能。

    Parameters
    ----------
    window : int
        看幾根來估動能（建議 10 ~ 30）
    scale  : float
        放大係數（1.0 = 原始動能；>1 = 誇大；<1 = 縮小）
    """
    if len(close_history) < window + 1:
        return 0.0
    recent = close_history[-(window + 1):]
    rets   = np.diff(np.log(recent))
    return float(np.mean(rets) * scale)


def run_simulation(
    df_real: pd.DataFrame,
    sim_bars: int = 200,
    warmup_bars: int = 100,
    impact_coeff: float = 0.001,
    intra_noise_scale: float = 1.0,
    drift_per_bar: float = 0.0,
    momentum_window: int = 20,
    momentum_scale: float = 1.0,
    bias_decay: float = 0.95,
    use_momentum_init: bool = False,
    n_institution: int = 5,
    n_momentum:    int = 40,
    n_random:      int = 100,
    n_contrarian:  int = 15,
    seed: int | None = 42,
    agents: list[BaseAgent] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    執行一次 ABM 模擬。

    Parameters
    ----------
    use_momentum_init : bool
        若為 True，自動從 warmup 最後 momentum_window 根估算動能 drift，
        注入 MomentumTrader 的 bias（exponential decay）。
        此時 drift_per_bar 的全局 drift 仍然獨立運作（建議維持 0.0）。
    momentum_window : int
        估算近期動能的窗口長度。
    momentum_scale : float
        動能強度放大係數。
    bias_decay : float
        MomentumTrader bias 的每根衰減係數。

    Returns
    -------
    df_warmup : 用作 context 的真實歷史（最後 warmup_bars 根）
    df_sim    : 模擬 OHLCV DataFrame
    """
    rng = np.random.default_rng(seed)

    df_ctx  = df_real.tail(warmup_bars).reset_index(drop=True)
    closes  = df_ctx["Close"].values.astype(float)
    highs   = df_ctx["High"].values.astype(float)
    lows    = df_ctx["Low"].values.astype(float)
    volumes = df_ctx["Volume"].values.astype(float)

    # 計算動能初始偏移
    momentum_bias = 0.0
    if use_momentum_init:
        momentum_bias = estimate_momentum_drift(
            closes, window=momentum_window, scale=momentum_scale
        )

    if agents is None:
        agents = build_default_agents(
            n_institution=n_institution,
            n_momentum=n_momentum,
            n_random=n_random,
            n_contrarian=n_contrarian,
            momentum_bias=momentum_bias,
            bias_decay=bias_decay,
            rng=rng,
        )
    else:
        # 重置 agent 狀態，防止上一次模擬的記憶污染
        for a in agents:
            a.reset_state()

    engine = MarketEngine(
        agents=agents,
        impact_coeff=impact_coeff,
        intra_noise_scale=intra_noise_scale,
        drift_per_bar=drift_per_bar,
        seed=int(rng.integers(0, 2**31)),
    )

    rows = []
    last_date = pd.Timestamp(df_ctx["Date"].iloc[-1])

    for i in range(sim_bars):
        bar = engine.step(
            close_history=closes,
            high_history=highs,
            low_history=lows,
            volume_history=volumes,
            bar_idx=warmup_bars + i,
        )
        next_date = last_date + pd.offsets.BDay(1)
        last_date = next_date
        rows.append({
            "Date":     next_date,
            "Open":     bar["open"],
            "High":     bar["high"],
            "Low":      bar["low"],
            "Close":    bar["close"],
            "Volume":   bar["volume"],
            "NetOrder": bar["net_order"],
        })
        closes  = np.append(closes,  bar["close"])
        highs   = np.append(highs,   bar["high"])
        lows    = np.append(lows,    bar["low"])
        volumes = np.append(volumes, bar["volume"])

    return df_ctx, pd.DataFrame(rows)
