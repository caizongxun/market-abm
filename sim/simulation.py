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

auto_drift 模式（預設開啟）
-----------------------------
當 use_momentum_init=True 且 drift_per_bar=0.0 時，自動把 momentum_bias
注入 drift_per_bar（逐 bar 歸零）而不是透過 agent 訂單量。
這樣能避免 bias 被 total_capital 稀釋成幾乎零。

  effective_drift[bar_i] = momentum_bias * decay^i

這個 drift 直接加進 new_open = last_close * exp(order_impact + drift[i])。

Rolling calibration
-------------------
見 sim/regime.py 的 RegimeCalibrator。
run_simulation_rolling() 是薄 wrapper，讓 run_sim.py 用 --rolling flag 呼叫。

Window 接縫對齊
---------------
initial_price 參數讓 MarketEngine 以指定價格作為第一根 bar 的 last_close，
消除 rolling window 拼接時的價格跳層。
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
    impact_coeff: float = 0.0015,
    intra_noise_scale: float = 1.0,
    drift_per_bar: float = 0.0,
    momentum_window_fast: int = 5,
    momentum_window_slow: int = 20,
    momentum_scale: float = 1.0,
    bias_decay: float = 0.97,
    use_momentum_init: bool = False,
    auto_drift: bool = True,
    n_institution: int = 5,
    n_momentum:    int = 40,
    n_random:      int = 100,
    n_contrarian:  int = 15,
    seed: int | None = 42,
    agents: list[BaseAgent] | None = None,
    path_floor_pct: float = 0.30,
    initial_price: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    執行一次 ABM 模擬。

    df_real 接受兩種格式：
      (a) DatetimeIndex + OHLCV 欄位（fetch.py 回傳的原始格式）
      (b) 整數 index + Date 欄位（舊快取檔或手動建立的 DataFrame）

    Parameters
    ----------
    auto_drift : bool
        當 True 且 use_momentum_init=True 且 drift_per_bar==0 時，
        自動把 momentum_bias 注入 drift schedule（逐 bar 歸零）。
    path_floor_pct : float
        路徑級別 floor：任何 bar 的 close 不得低於起始 close * (1 - path_floor_pct)。
    initial_price : float | None
        若指定，MarketEngine 以此價格作為第一根 bar 的 last_close，
        覆蓋 close_history[-1]。用於 rolling window 接縫對齊：
        傳入前一個 window 最後一根真實收盤價，確保模擬路徑從正確水位出發。

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

    start_close    = float(closes[-1])
    path_min_price = start_close * (1.0 - path_floor_pct) if path_floor_pct > 0 else 0.0

    # --- 取最後一根 bar 的日期，支援兩種載入格式 ---
    _tail = df_real.tail(1)
    if isinstance(_tail.index, pd.DatetimeIndex):
        last_date = pd.Timestamp(_tail.index[-1])
    elif "Date" in df_real.columns:
        last_date = pd.Timestamp(df_real["Date"].iloc[-1])
    else:
        last_date = pd.Timestamp("2000-01-01")  # fallback

    # Momentum bias
    momentum_bias   = 0.0
    momentum_reason = "off"
    if use_momentum_init:
        momentum_bias, momentum_reason = estimate_momentum_drift_dual(
            closes,
            window_fast=momentum_window_fast,
            window_slow=momentum_window_slow,
            scale=momentum_scale,
        )

    # Auto-drift: convert momentum_bias -> per-bar drift schedule
    drift_schedule: np.ndarray | None = None
    if use_momentum_init and auto_drift and drift_per_bar == 0.0 and momentum_bias != 0.0:
        drift_schedule = momentum_bias * (bias_decay ** np.arange(sim_bars))

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
        initial_price=initial_price,   # 接縫對齊
    )

    rows = []

    for i in range(sim_bars):
        if drift_schedule is not None:
            engine.drift_per_bar = float(drift_schedule[i])

        bar = engine.step(
            close_history=closes,
            high_history=highs,
            low_history=lows,
            volume_history=volumes,
            bar_idx=warmup_bars + i,
        )

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


# ---------------------------------------------------------------------------
# Rolling wrapper (thin, delegates to RegimeCalibrator)
# ---------------------------------------------------------------------------
def run_simulation_rolling(
    df_all: pd.DataFrame,
    lookback: int = 60,
    step: int = 20,
    n_sims: int = 10,
    warmup_bars: int = 60,
    ema_alpha: float = 0.4,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Thin wrapper around RegimeCalibrator.run().
    回傳拼接後的模擬 DataFrame 和每個 window 的參數記錄。
    """
    from .regime import RegimeCalibrator
    cal = RegimeCalibrator(
        lookback=lookback,
        step=step,
        n_sims=n_sims,
        warmup_bars=warmup_bars,
        ema_alpha=ema_alpha,
        verbose=verbose,
    )
    return cal.run(df_all, seed=seed)
