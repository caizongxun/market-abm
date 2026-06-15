"""
simulation.py
=============
主迴圈：以真實歷史 K 棒作為初始條件，
讓 agent 群往後模擬 N 根，產生模擬 K 棒序列。

流程
----
1. 讀入真實 OHLCV（作為 warm-up context）
2. 從指定起始點開始，每輪呼叫 MarketEngine.step()
3. 新產生的 K 棒立刻加入歷史（rolling window）
4. 回傳模擬 K 棒 DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .agents import BaseAgent, build_default_agents
from .market import MarketEngine


def run_simulation(
    df_real: pd.DataFrame,
    sim_bars: int = 200,
    warmup_bars: int = 100,
    impact_coeff: float = 0.0005,
    intra_noise_scale: float = 0.6,
    n_institution: int = 5,
    n_momentum:    int = 40,
    n_random:      int = 100,
    n_contrarian:  int = 15,
    seed: int | None = 42,
    agents: list[BaseAgent] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    執行模擬。

    Parameters
    ----------
    df_real        : 真實歷史 OHLCV，需含欄位 Date/Open/High/Low/Close/Volume
    sim_bars       : 要模擬的 K 棒根數
    warmup_bars    : 用來初始化 agent 狀態的歷史根數（不進入模擬輸出）
    impact_coeff   : 市場衝擊係數
    intra_noise_scale : K 棒內波動 ATR 倍數
    seed           : 隨機種子
    agents         : 自訂 agent 列表；None 則使用預設組合

    Returns
    -------
    df_warmup : 用作 context 的真實歷史（最後 warmup_bars 根）
    df_sim    : 模擬產生的 K 棒 DataFrame
    """
    rng = np.random.default_rng(seed)

    if agents is None:
        agents = build_default_agents(
            n_institution=n_institution,
            n_momentum=n_momentum,
            n_random=n_random,
            n_contrarian=n_contrarian,
            rng=rng,
        )

    engine = MarketEngine(
        agents=agents,
        impact_coeff=impact_coeff,
        intra_noise_scale=intra_noise_scale,
        seed=int(rng.integers(0, 2**31)),
    )

    # 準備 warm-up 歷史
    df_ctx  = df_real.tail(warmup_bars).reset_index(drop=True)
    closes  = df_ctx["Close"].values.astype(float)
    highs   = df_ctx["High"].values.astype(float)
    lows    = df_ctx["Low"].values.astype(float)
    volumes = df_ctx["Volume"].values.astype(float)

    rows = []
    # 推算模擬起始日期（warm-up 最後一根之後）
    last_date = pd.Timestamp(df_ctx["Date"].iloc[-1])

    for i in range(sim_bars):
        bar = engine.step(
            close_history=closes,
            high_history=highs,
            low_history=lows,
            volume_history=volumes,
            bar_idx=warmup_bars + i,
        )

        # 推算下一個交易日
        next_date = last_date + pd.offsets.BDay(1)
        last_date = next_date

        rows.append({
            "Date":      next_date,
            "Open":      bar["open"],
            "High":      bar["high"],
            "Low":       bar["low"],
            "Close":     bar["close"],
            "Volume":    bar["volume"],
            "NetOrder":  bar["net_order"],
        })

        # rolling：新 K 棒加入歷史
        closes  = np.append(closes,  bar["close"])
        highs   = np.append(highs,   bar["high"])
        lows    = np.append(lows,    bar["low"])
        volumes = np.append(volumes, bar["volume"])

    df_sim = pd.DataFrame(rows)
    return df_ctx, df_sim
