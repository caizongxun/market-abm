"""
stat_process.py
===============
純統計過程模型，完全不使用 agent。

Pipeline
--------
1. fit(df_history)
   從歷史 K 棒擬合 5 個參數：
     ret_mu       : 對數報酬均值
     ret_sigma    : 對數報酬標準差
     ret_df       : Student-t 自由度（控制 kurtosis / 肥尾）
     hurst_target : Hurst 指數（控制方向記憶）
     wick_lambda  : 影線長度的 Exponential 分佈 scale（單位：ATR 倍數）

2. generate(params, n_bars, seed)
   用擬合好的參數取樣 n_bars 根 K 棒：
     - 對數報酬 ~ t(df, mu, sigma)
     - 方向記憶注入（FGN 近似，基於 Hurst 的 AR(1) 代理）
     - OHLC 重建：Open = prev_Close, Close = Open * exp(r)
     - High/Low：body ± wick，wick ~ Exp(lambda * ATR)

3. rolling_fit_generate(df_real, lookback, step, n_forward, seed)
   Rolling 模式：每隔 step 根 K 棒重新擬合，往前生成 n_forward 根模擬棒。
   回傳整段模擬 DataFrame，供 metrics.compare() 比較。

用法
----
  from sim.stat_process import fit, generate, rolling_fit_generate
"""

from __future__ import annotations

import warnings
from typing import TypedDict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from sim.metrics import hurst_exponent


# ─────────────────────────────────────────────────────────────────────────────
# 型別
# ─────────────────────────────────────────────────────────────────────────────

class StatParams(TypedDict):
    ret_mu:       float   # 對數報酬均值
    ret_sigma:    float   # 對數報酬標準差
    ret_df:       float   # Student-t 自由度（2.1 ~ 30）
    hurst_target: float   # Hurst 指數（0.3 ~ 0.8）
    wick_lambda:  float   # 影線 Exp scale，單位 = ATR 倍數


# ─────────────────────────────────────────────────────────────────────────────
# 1. FIT
# ─────────────────────────────────────────────────────────────────────────────

def _log_returns(closes: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.maximum(closes.astype(float), 1e-10)))


def _fit_student_t(log_rets: np.ndarray) -> tuple[float, float, float]:
    """MLE 擬合 Student-t，回傳 (df, mu, sigma)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df, mu, sigma = stats.t.fit(log_rets, floc=None, fscale=None)
            df    = float(np.clip(df, 2.1, 30.0))
            sigma = float(np.maximum(sigma, 1e-6))
            mu    = float(mu)
        except Exception:
            mu, sigma = float(np.mean(log_rets)), float(np.std(log_rets))
            df = 10.0
    return df, mu, sigma


def _fit_wick_lambda(df_ohlc: pd.DataFrame) -> float:
    """
    計算上下影線長度 / ATR，用 Exponential MLE 估算 scale。
    wick_lambda = E[wick / ATR]
    """
    hi   = df_ohlc["High"].values.astype(float)
    lo   = df_ohlc["Low"].values.astype(float)
    op   = df_ohlc["Open"].values.astype(float)
    cl   = df_ohlc["Close"].values.astype(float)

    body_hi  = np.maximum(op, cl)
    body_lo  = np.minimum(op, cl)
    upper_wick = np.maximum(hi - body_hi, 0.0)
    lower_wick = np.maximum(body_lo - lo,  0.0)

    # ATR（簡化：High - Low，避免需要前一根 Close）
    atr = hi - lo
    atr = np.maximum(atr, 1e-10)

    wick_ratio = np.concatenate([upper_wick / atr, lower_wick / atr])
    wick_ratio = wick_ratio[wick_ratio > 0]

    if len(wick_ratio) < 10:
        return 0.3  # fallback

    # Exponential MLE：lambda = mean
    return float(np.mean(wick_ratio))


def fit(df_history: pd.DataFrame) -> StatParams:
    """
    從歷史 K 棒 DataFrame（需含 Open/High/Low/Close）擬合 5 個參數。

    Parameters
    ----------
    df_history : pd.DataFrame
        必須包含欄位 Open, High, Low, Close。

    Returns
    -------
    StatParams
    """
    closes   = df_history["Close"].values
    log_rets = _log_returns(closes)

    if len(log_rets) < 5:
        raise ValueError(f"lookback 太短（{len(log_rets)} bars），需要至少 5 根。")

    df_t, mu, sigma = _fit_student_t(log_rets)
    h               = hurst_exponent(log_rets)
    wick_lam        = _fit_wick_lambda(df_history)

    return StatParams(
        ret_mu       = mu,
        ret_sigma    = sigma,
        ret_df       = df_t,
        hurst_target = float(np.clip(h, 0.3, 0.8)),
        wick_lambda  = wick_lam,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE
# ─────────────────────────────────────────────────────────────────────────────

def _ar1_hurst_rho(h: float) -> float:
    """
    用 AR(1) 代理 FGN：給定目標 Hurst H，反解 AR(1) 的 rho。
    近似公式：rho ≈ 2^(2H-1) - 1
    H=0.5 → rho=0，H>0.5 → rho>0（趨勢），H<0.5 → rho<0（均值回歸）
    """
    rho = 2 ** (2 * h - 1) - 1
    return float(np.clip(rho, -0.95, 0.95))


def generate(
    params:       StatParams,
    n_bars:       int,
    start_price:  float  = 100.0,
    seed:         int | None = None,
) -> pd.DataFrame:
    """
    用 StatParams 取樣 n_bars 根合成 K 棒。

    Parameters
    ----------
    params      : 由 fit() 回傳的參數字典
    n_bars      : 要生成的 K 棒數量
    start_price : 起始 Open 價格
    seed        : 亂數種子

    Returns
    -------
    pd.DataFrame，欄位：Open, High, Low, Close, Volume（dummy）
    """
    rng = np.random.default_rng(seed)

    mu      = params["ret_mu"]
    sigma   = params["ret_sigma"]
    df_t    = params["ret_df"]
    hurst   = params["hurst_target"]
    wick_lam= params["wick_lambda"]

    # --- 取樣對數報酬（Student-t）---
    # scipy t 的參數：(df, loc=mu, scale=sigma)
    raw_rets = stats.t.rvs(df=df_t, loc=mu, scale=sigma, size=n_bars, random_state=rng)

    # --- 注入方向記憶（AR(1) 代理 FGN）---
    rho = _ar1_hurst_rho(hurst)
    if abs(rho) > 0.01:
        mem_rets = np.empty(n_bars)
        mem_rets[0] = raw_rets[0]
        innov_scale = np.sqrt(1 - rho ** 2)
        for i in range(1, n_bars):
            mem_rets[i] = rho * mem_rets[i-1] + innov_scale * raw_rets[i]
        # 還原原本的 mu/sigma（AR(1) 不改變邊際分佈均值，但會改 scale）
        cur_std = np.std(mem_rets)
        if cur_std > 1e-10:
            mem_rets = (mem_rets - np.mean(mem_rets)) / cur_std * sigma + mu
        log_rets = mem_rets
    else:
        log_rets = raw_rets

    # --- 重建 OHLC ---
    opens  = np.empty(n_bars)
    closes = np.empty(n_bars)
    highs  = np.empty(n_bars)
    lows   = np.empty(n_bars)

    opens[0] = start_price
    for i in range(n_bars):
        if i > 0:
            opens[i] = closes[i - 1]
        closes[i] = opens[i] * np.exp(log_rets[i])

    # --- 影線（wick ~ Exp(lambda) * ATR_proxy）---
    body_size = np.abs(closes - opens)
    atr_proxy = np.maximum(body_size, np.mean(body_size) * 0.5)

    upper_wicks = rng.exponential(scale=wick_lam * atr_proxy)
    lower_wicks = rng.exponential(scale=wick_lam * atr_proxy)

    body_hi = np.maximum(opens, closes)
    body_lo = np.minimum(opens, closes)
    highs   = body_hi + upper_wicks
    lows    = body_lo - lower_wicks

    # 成交量：dummy，用 log-normal 取樣，保持視覺合理
    volumes = rng.lognormal(mean=15.0, sigma=0.5, size=n_bars).astype(int)

    df_sim = pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    })

    return df_sim


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROLLING FIT → GENERATE
# ─────────────────────────────────────────────────────────────────────────────

def rolling_fit_generate(
    df_real:    pd.DataFrame,
    lookback:   int  = 60,
    step:       int  = 20,
    n_forward:  int  | None = None,   # None → 等於 step
    seed:       int  = 42,
    verbose:    bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Rolling 模式：每隔 step 根重新擬合，往前生成模擬 K 棒。

    Parameters
    ----------
    df_real   : 完整歷史 K 棒
    lookback  : 每次擬合用的歷史長度
    step      : 每次前進的 K 棒數
    n_forward : 每次生成的模擬棒數（None = step）
    seed      : 亂數種子基底
    verbose   : 是否印出每個 window 的參數

    Returns
    -------
    (df_sim_all, param_log)
      df_sim_all : 拼接後的完整模擬 DataFrame
      param_log  : list of dict，每個 window 的擬合參數 + loss
    """
    if n_forward is None:
        n_forward = step

    n_total = len(df_real)
    sim_chunks: list[pd.DataFrame] = []
    param_log:  list[dict]         = []

    window_idx = 0
    pos = lookback  # 第一個擬合窗口的終點

    while pos <= n_total:
        fit_start = pos - lookback
        fit_end   = pos
        fwd_end   = min(pos + n_forward, n_total)
        actual_fwd = fwd_end - pos

        if actual_fwd <= 0:
            break

        df_window = df_real.iloc[fit_start:fit_end].copy().reset_index(drop=True)
        params    = fit(df_window)

        # 用前一窗口的最後 Close 作為起始價格
        start_px = float(df_real["Close"].iloc[fit_end - 1])

        df_chunk = generate(
            params      = params,
            n_bars      = actual_fwd,
            start_price = start_px,
            seed        = seed + window_idx,
        )
        sim_chunks.append(df_chunk)

        # 計算 loss：vol_err + kurt_err
        real_fwd = df_real.iloc[pos:fwd_end].copy().reset_index(drop=True)
        real_rets = np.diff(np.log(np.maximum(real_fwd["Close"].values, 1e-10)))
        sim_rets  = np.diff(np.log(np.maximum(df_chunk["Close"].values, 1e-10)))
        if len(real_rets) > 1 and len(sim_rets) > 1:
            vol_err  = abs(np.std(sim_rets) - np.std(real_rets)) / max(np.std(real_rets), 1e-8)
            kurt_err = abs(float(stats.kurtosis(sim_rets)) - float(stats.kurtosis(real_rets)))
            loss = vol_err * 3.0 + kurt_err * 0.5
        else:
            loss = 0.0

        log_entry = {
            "window":       window_idx + 1,
            "fit_bars":     [fit_start, fit_end],
            "fwd_bars":     [pos, fwd_end],
            **params,
            "loss":         round(loss, 4),
        }
        param_log.append(log_entry)

        if verbose:
            print(
                f"[stat] window {window_idx+1:>3}  "
                f"fit=[{fit_start}:{fit_end}]  fwd=[{pos}:{fwd_end}]  "
                f"df={params['ret_df']:.1f}  "
                f"sigma={params['ret_sigma']:.4f}  "
                f"hurst={params['hurst_target']:.3f}  "
                f"wick={params['wick_lambda']:.3f}  "
                f"loss={loss:.4f}"
            )

        pos += step
        window_idx += 1

    if not sim_chunks:
        raise RuntimeError("沒有產生任何模擬 chunk，請檢查 lookback/step 設定。")

    df_sim_all = pd.concat(sim_chunks, ignore_index=True)
    return df_sim_all, param_log
