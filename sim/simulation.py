"""
simulation.py
=============
主迴圈：用真實歷史 K 棒初始化 agent 狀態，然後模擬 N 根 K 棒。

Momentum-init 策略（雙窗口）
-----------------------------
短窗口（fast, 預設 5 根）抓轉折敏感度。
長窗口（slow, 預設 20 根）確認趨勢一致性。

決策邏輯：
  - fast 與 slow 同向 → 使用 fast drift（趨勢確立）
  - fast 與 slow 反向 → 使用 fast drift（已發生反轉，以短期為準）
  - fast 接近 0       → 不注入偏移（市場橫盤）

這樣無論是趨勢延續還是趨勢反轉，都能用最新的方向信號。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .agents import BaseAgent, build_default_agents
from .market import MarketEngine


def estimate_momentum_drift(
    close_history: np.ndarray,
    window: int = 5,
    scale: float = 1.0,
) -> float:
    """
    從最近 window 根收盤價估算動能 drift（平均每根 log-return）。

    Parameters
    ----------
    window : int   建議 3~10（短窗口抓轉折）
    scale  : float 放大係數
    """
    if len(close_history) < window + 1:
        return 0.0
    recent = close_history[-(window + 1):]
    rets   = np.diff(np.log(recent))
    return float(np.mean(rets) * scale)


def estimate_momentum_drift_dual(
    close_history: np.ndarray,
    window_fast:  int   = 5,
    window_slow:  int   = 20,
    scale:        float = 1.0,
    flat_threshold: float = 1e-5,
) -> tuple[float, str]:
    """
    雙窗口動能估算，回傳最終使用的 drift 和決策理由。

    決策邏輯
    --------
    1. fast drift 接近 0（< flat_threshold）→ 橫盤，不注入偏移，回傳 0
    2. fast 與 slow 同向 → 趨勢確立，使用 fast drift
    3. fast 與 slow 反向 → 趨勢反轉，以 fast 為準（更新鮮）

    Returns
    -------
    (drift, reason_str)
    """
    drift_fast = estimate_momentum_drift(close_history, window=window_fast, scale=scale)
    drift_slow = estimate_momentum_drift(close_history, window=window_slow, scale=scale)

    if abs(drift_fast) < flat_threshold:
        return 0.0, f"flat (fast={drift_fast:+.6f}, slow={drift_slow:+.6f})"

    same_direction = (drift_fast > 0) == (drift_slow > 0)
    if same_direction:
        reason = (f"trend confirmed (fast={drift_fast:+.6f}, "
                  f"slow={drift_slow:+.6f}) → using fast")
    else:
        reason = (f"REVERSAL detected (fast={drift_fast:+.6f}, "
                  f"slow={drift_slow:+.6f}) → using fast (more recent)")

    return drift_fast, reason


def run_simulation(
    df_real: pd.DataFrame,
    sim_bars: int = 200,
    warmup_bars: int = 100,
    impact_coeff: float = 0.001,
    intra_noise_scale: float = 1.0,
    drift_per_bar: float = 0.0,
    momentum_window_fast: int = 5,
    momentum_window_slow: int = 20,
    momentum_scale: float = 1.0,
    bias_decay: float = 0.95,
    use_momentum_init: bool = False,
    n_institution: int = 5,
    n_momentum:    int = 40,
    n_random:      int = 100,
    n_contrarian:  int = 15,
    seed: int | None = 42,
    agents: list[BaseAgent] | None = None,
    path_floor_pct: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    執行一次 ABM 模擬。

    Parameters
    ----------
    momentum_window_fast : int
        短窗口（抓轉折），預設 5 根。
    momentum_window_slow : int
        長窗口（確認趨勢），預設 20 根。
    path_floor_pct : float
        路徑級別 floor：任何 bar 的 close 不得低於起始 close * (1 - path_floor_pct)。
        預設 0.30，即不得跌超過 30%。設 0 停用。

    Returns
    -------
    df_warmup, df_sim
    """
    rng = np.random.default_rng(seed)

    df_ctx  = df_real.tail(warmup_bars).reset_index(drop=True)
    closes  = df_ctx["Close"].values.astype(float)
    highs   = df_ctx["High"].values.astype(float)
    lows    = df_ctx["Low"].values.astype(float)
    volumes = df_ctx["Volume"].values.astype(float)

    # 路徑起始價格 floor
    start_close   = float(closes[-1])
    path_min_price = start_close * (1.0 - path_floor_pct) if path_floor_pct > 0 else 0.0

    # 動能初始偏移
    momentum_bias  = 0.0
    momentum_reason = "off"
    if use_momentum_init:
        momentum_bias, momentum_reason = estimate_momentum_drift_dual(
            closes,
            window_fast=momentum_window_fast,
            window_slow=momentum_window_slow,
            scale=momentum_scale,
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
        for a in agents:
            a.reset_state()

    engine = MarketEngine(
        agents=agents,
        impact_coeff=impact_coeff,
        intra_noise_scale=intra_noise_scale,
        drift_per_bar=drift_per_bar,
        seed=int(rng.integers(0, 2**31)),
    )

    rows      = []
    last_date = pd.Timestamp(df_ctx["Date"].iloc[-1])

    for i in range(sim_bars):
        bar = engine.step(
            close_history=closes,
            high_history=highs,
            low_history=lows,
            volume_history=volumes,
            bar_idx=warmup_bars + i,
        )

        # Path-level floor：防止連續 floor 導致路徑崩潰
        if path_min_price > 0 and bar["close"] < path_min_price:
            bar["close"] = path_min_price
            bar["low"]   = min(bar["low"], path_min_price)

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
