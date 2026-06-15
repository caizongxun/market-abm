"""
metrics.py
==========
統計驗證：比較模擬 K 棒與真實 K 棒的統計特性。

指標
----
1. 日報酬分佈：均值 / 標準差 / 偏度 / 峰度
2. Hurst 指數（趨勢慣性）
3. 波動率自相關（GARCH 效果代理）
4. 方向命中率（模擬 vs. 真實的方向一致性）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────────────────────

def log_returns(closes: np.ndarray) -> np.ndarray:
    return np.diff(np.log(closes.astype(float) + 1e-10))


def hurst_exponent(series: np.ndarray, min_lag: int = 2, max_lag: int = 50) -> float:
    """
    R/S 法估算 Hurst 指數。
    H > 0.5 = 趨勢；H < 0.5 = 均值回歸；H ≈ 0.5 = 隨機遊走。
    """
    lags   = range(min_lag, min(max_lag, len(series) // 4))
    tau    = []
    rs_arr = []
    for lag in lags:
        chunks = [series[i:i+lag] for i in range(0, len(series) - lag, lag)]
        rs_vals = []
        for c in chunks:
            if len(c) < 2:
                continue
            m   = np.mean(c)
            dev = np.cumsum(c - m)
            r   = np.ptp(dev)
            s   = np.std(c, ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            tau.append(lag)
            rs_arr.append(np.mean(rs_vals))

    if len(tau) < 2:
        return 0.5
    log_tau = np.log(tau)
    log_rs  = np.log(rs_arr)
    slope, *_ = np.polyfit(log_tau, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))


def vol_autocorr(log_rets: np.ndarray, lags: int = 10) -> np.ndarray:
    """絕對報酬的自相關（GARCH 效果代理）。"""
    abs_rets = np.abs(log_rets)
    result = []
    for lag in range(1, lags + 1):
        if len(abs_rets) <= lag:
            result.append(0.0)
        else:
            c = np.corrcoef(abs_rets[:-lag], abs_rets[lag:])[0, 1]
            result.append(float(c) if not np.isnan(c) else 0.0)
    return np.array(result)


# ─────────────────────────────────────────────────────────────────────────────
# 主要對比函式
# ─────────────────────────────────────────────────────────────────────────────

def compare(
    df_real: pd.DataFrame,
    df_sim:  pd.DataFrame,
    print_report: bool = True,
) -> dict:
    """
    輸入真實 K 棒和模擬 K 棒（長度可不同），
    計算並比較各項統計指標。
    """
    real_rets = log_returns(df_real["Close"].values)
    sim_rets  = log_returns(df_sim["Close"].values)

    def dist_stats(r):
        return {
            "mean":     float(np.mean(r)),
            "std":      float(np.std(r)),
            "skew":     float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r)),   # excess kurtosis
        }

    real_stats = dist_stats(real_rets)
    sim_stats  = dist_stats(sim_rets)

    real_hurst = hurst_exponent(real_rets)
    sim_hurst  = hurst_exponent(sim_rets)

    real_autocorr = vol_autocorr(real_rets)
    sim_autocorr  = vol_autocorr(sim_rets)

    # 方向命中率（模擬序列對齊長度）
    min_len = min(len(real_rets), len(sim_rets))
    direction_hit = float(np.mean(
        np.sign(real_rets[:min_len]) == np.sign(sim_rets[:min_len])
    )) if min_len > 0 else 0.0

    result = {
        "real": {**real_stats, "hurst": real_hurst, "vol_autocorr": real_autocorr.tolist()},
        "sim":  {**sim_stats,  "hurst": sim_hurst,  "vol_autocorr": sim_autocorr.tolist()},
        "direction_hit_rate": direction_hit,
    }

    if print_report:
        _print_report(result)

    return result


def _print_report(r: dict):
    real, sim = r["real"], r["sim"]
    print("\n" + "="*52)
    print(f"  {'指標':<20} {'真實市場':>12} {'模擬市場':>12}")
    print("="*52)
    for key in ["mean", "std", "skew", "kurtosis", "hurst"]:
        print(f"  {key:<20} {real[key]:>12.5f} {sim[key]:>12.5f}")
    print("-"*52)
    print(f"  方向命中率           {r['direction_hit_rate']:>12.4f}")
    print("="*52)
    print()
