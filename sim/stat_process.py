"""
stat_process.py  v2
===================
純統計過程模型，完全不使用 agent。

v2 修正
-------
  Fix-1 : Student-t df 掃描 log-likelihood，不再鎖死在 clip 上限
  Fix-2 : 改用 skewnorm 擬合，保留 skew 方向
  Fix-3 : wick_lambda 使用真實 rolling ATR（Wilder，14 根），不用 body_size

Pipeline
--------
1. fit(df_history)         → StatParams (7 個參數)
2. generate(params, n_bars, seed)  → OHLC DataFrame
3. rolling_fit_generate(df_real, lookback, step, ...)  → (df_sim, param_log)

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
    ret_mu:       float   # skewnorm loc
    ret_sigma:    float   # skewnorm scale
    ret_skew_a:   float   # skewnorm shape (alpha)
    ret_df:       float   # Student-t df，用於尾巴厚度估計（僅診斷用）
    hurst_target: float   # Hurst 指數（0.3 ~ 0.8）
    wick_lambda:  float   # 影線 Exp scale，單位 = ATR 倍數
    atr_mean:     float   # 歷史平均 ATR（絕對值，用於 wick 生成）


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def _log_returns(closes: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.maximum(closes.astype(float), 1e-10)))


def _wilder_atr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder 平滑 ATR；回傳與輸入等長的 array（前 period-1 根用簡單均值填充）。"""
    n = len(hi)
    tr = np.empty(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))

    atr = np.empty(n)
    # 前 period 根用簡單均值
    atr[:period] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


# ─────────────────────────────────────────────────────────────────────────────
# Fix-1 + Fix-2 : 擬合 skewnorm + 掃描 df
# ─────────────────────────────────────────────────────────────────────────────

def _fit_df_scan(log_rets: np.ndarray) -> float:
    """
    Fix-1: 對 df ∈ [2.1, 30] 做 log-likelihood 掃描，找最佳 Student-t df。
    用對稱 t（忽略 skew）只是為了量化尾巴厚度，不作為取樣分佈使用。
    """
    mu_hat    = float(np.mean(log_rets))
    sigma_hat = float(np.std(log_rets, ddof=1))

    def neg_ll(df):
        return -np.sum(stats.t.logpdf(log_rets, df=df, loc=mu_hat, scale=sigma_hat))

    result = minimize_scalar(neg_ll, bounds=(2.1, 30.0), method="bounded")
    return float(result.x)


def _fit_skewnorm(log_rets: np.ndarray) -> tuple[float, float, float]:
    """
    Fix-2: MLE 擬合 skewnorm，回傳 (skew_a, mu, sigma)。
    保留分佈的偏度方向（AAPL 真實 skew > 0 ↔ skew_a > 0）。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            a, loc, scale = stats.skewnorm.fit(log_rets)
            a     = float(np.clip(a, -10.0, 10.0))
            scale = float(np.maximum(scale, 1e-6))
            loc   = float(loc)
        except Exception:
            a, loc, scale = 0.0, float(np.mean(log_rets)), float(np.std(log_rets))
    return a, loc, scale


# ─────────────────────────────────────────────────────────────────────────────
# Fix-3 : wick_lambda 使用 rolling ATR
# ─────────────────────────────────────────────────────────────────────────────

def _fit_wick_lambda(df_ohlc: pd.DataFrame) -> tuple[float, float]:
    """
    Fix-3: 用 Wilder ATR 作為正規化基準，回傳 (wick_lambda, atr_mean)。
    wick_lambda = E[(upper_wick + lower_wick) / ATR]
    """
    hi = df_ohlc["High"].values.astype(float)
    lo = df_ohlc["Low"].values.astype(float)
    op = df_ohlc["Open"].values.astype(float)
    cl = df_ohlc["Close"].values.astype(float)

    body_hi    = np.maximum(op, cl)
    body_lo    = np.minimum(op, cl)
    upper_wick = np.maximum(hi - body_hi, 0.0)
    lower_wick = np.maximum(body_lo - lo, 0.0)

    atr = _wilder_atr(hi, lo, cl, period=14)
    atr = np.maximum(atr, 1e-10)
    atr_mean = float(np.mean(atr))

    wick_ratio = np.concatenate([upper_wick / atr, lower_wick / atr])
    wick_ratio = wick_ratio[wick_ratio > 0]

    if len(wick_ratio) < 10:
        return 0.3, atr_mean

    return float(np.mean(wick_ratio)), atr_mean


# ─────────────────────────────────────────────────────────────────────────────
# 1. FIT
# ─────────────────────────────────────────────────────────────────────────────

def fit(df_history: pd.DataFrame) -> StatParams:
    """
    從歷史 K 棒 DataFrame（需含 Open/High/Low/Close）擬合 7 個參數。

    Returns
    -------
    StatParams
    """
    closes   = df_history["Close"].values
    log_rets = _log_returns(closes)

    if len(log_rets) < 5:
        raise ValueError(f"lookback 太短（{len(log_rets)} bars），需要至少 5 根。")

    # Fix-1: df 掃描
    df_t = _fit_df_scan(log_rets)

    # Fix-2: skewnorm
    skew_a, mu, sigma = _fit_skewnorm(log_rets)

    # Hurst
    h = hurst_exponent(log_rets)

    # Fix-3: rolling ATR
    wick_lam, atr_mean = _fit_wick_lambda(df_history)

    return StatParams(
        ret_mu       = mu,
        ret_sigma    = sigma,
        ret_skew_a   = skew_a,
        ret_df       = df_t,
        hurst_target = float(np.clip(h, 0.3, 0.8)),
        wick_lambda  = wick_lam,
        atr_mean     = atr_mean,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE
# ─────────────────────────────────────────────────────────────────────────────

def _ar1_hurst_rho(h: float) -> float:
    rho = 2 ** (2 * h - 1) - 1
    return float(np.clip(rho, -0.95, 0.95))


def generate(
    params:       StatParams,
    n_bars:       int,
    start_price:  float = 100.0,
    seed:         int | None = None,
) -> pd.DataFrame:
    """
    用 StatParams 取樣 n_bars 根合成 K 棒。

    Returns
    -------
    pd.DataFrame，欄位：Open, High, Low, Close, Volume（dummy）
    """
    rng = np.random.default_rng(seed)

    mu       = params["ret_mu"]
    sigma    = params["ret_sigma"]
    skew_a   = params["ret_skew_a"]
    hurst    = params["hurst_target"]
    wick_lam = params["wick_lambda"]
    atr_mean = params["atr_mean"]

    # --- Fix-2: 用 skewnorm 取樣，保留偏度 ---
    raw_rets = stats.skewnorm.rvs(
        a=skew_a, loc=mu, scale=sigma, size=n_bars, random_state=rng
    )

    # --- AR(1) 方向記憶注入 ---
    rho = _ar1_hurst_rho(hurst)
    if abs(rho) > 0.01:
        mem_rets = np.empty(n_bars)
        mem_rets[0] = raw_rets[0]
        innov_scale = np.sqrt(1 - rho ** 2)
        for i in range(1, n_bars):
            mem_rets[i] = rho * mem_rets[i-1] + innov_scale * raw_rets[i]
        # 還原 mu / sigma（AR(1) 改變 scale）
        cur_std = np.std(mem_rets)
        if cur_std > 1e-10:
            mem_rets = (mem_rets - np.mean(mem_rets)) / cur_std * sigma + mu
        log_rets = mem_rets
    else:
        log_rets = raw_rets

    # --- 重建 OHLC ---
    opens  = np.empty(n_bars)
    closes = np.empty(n_bars)

    opens[0] = start_price
    for i in range(n_bars):
        if i > 0:
            opens[i] = closes[i - 1]
        closes[i] = opens[i] * np.exp(log_rets[i])

    # --- Fix-3: wick 用 atr_mean 作為基準 scale ---
    # atr_proxy = 歷史平均 ATR（絕對值），保持與原始市場同量級
    atr_proxy = np.full(n_bars, atr_mean)

    upper_wicks = rng.exponential(scale=wick_lam * atr_proxy)
    lower_wicks = rng.exponential(scale=wick_lam * atr_proxy)

    body_hi = np.maximum(opens, closes)
    body_lo = np.minimum(opens, closes)
    highs   = body_hi + upper_wicks
    lows    = body_lo - lower_wicks

    volumes = rng.lognormal(mean=15.0, sigma=0.5, size=n_bars).astype(int)

    return pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROLLING FIT → GENERATE
# ─────────────────────────────────────────────────────────────────────────────

def rolling_fit_generate(
    df_real:   pd.DataFrame,
    lookback:  int = 60,
    step:      int = 20,
    n_forward: int | None = None,
    seed:      int = 42,
    verbose:   bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Rolling 模式：每隔 step 根重新擬合，往前生成模擬 K 棒。

    Returns
    -------
    (df_sim_all, param_log)
    """
    if n_forward is None:
        n_forward = step

    n_total    = len(df_real)
    sim_chunks: list[pd.DataFrame] = []
    param_log:  list[dict]         = []

    window_idx = 0
    pos        = lookback

    while pos <= n_total:
        fit_start  = pos - lookback
        fit_end    = pos
        fwd_end    = min(pos + n_forward, n_total)
        actual_fwd = fwd_end - pos

        if actual_fwd <= 0:
            break

        df_window = df_real.iloc[fit_start:fit_end].copy().reset_index(drop=True)
        params    = fit(df_window)
        start_px  = float(df_real["Close"].iloc[fit_end - 1])

        df_chunk = generate(
            params=params, n_bars=actual_fwd,
            start_price=start_px, seed=seed + window_idx,
        )
        sim_chunks.append(df_chunk)

        # Loss
        real_fwd  = df_real.iloc[pos:fwd_end].copy().reset_index(drop=True)
        real_rets = np.diff(np.log(np.maximum(real_fwd["Close"].values, 1e-10)))
        sim_rets  = np.diff(np.log(np.maximum(df_chunk["Close"].values, 1e-10)))
        if len(real_rets) > 1 and len(sim_rets) > 1:
            vol_err  = abs(np.std(sim_rets) - np.std(real_rets)) / max(np.std(real_rets), 1e-8)
            kurt_err = abs(float(stats.kurtosis(sim_rets)) - float(stats.kurtosis(real_rets)))
            skew_err = abs(float(stats.skew(sim_rets)) - float(stats.skew(real_rets)))
            loss     = vol_err * 3.0 + kurt_err * 0.5 + skew_err * 1.0
        else:
            loss = 0.0

        log_entry = {
            "window":   window_idx + 1,
            "fit_bars": [fit_start, fit_end],
            "fwd_bars": [pos, fwd_end],
            **params,
            "loss":     round(loss, 4),
        }
        param_log.append(log_entry)

        if verbose:
            print(
                f"[stat] window {window_idx+1:>3}  "
                f"fit=[{fit_start}:{fit_end}]  fwd=[{pos}:{fwd_end}]  "
                f"df={params['ret_df']:.2f}  "
                f"skew_a={params['ret_skew_a']:+.3f}  "
                f"sigma={params['ret_sigma']:.4f}  "
                f"hurst={params['hurst_target']:.3f}  "
                f"wick={params['wick_lambda']:.3f}  "
                f"loss={loss:.4f}"
            )

        pos += step
        window_idx += 1

    if not sim_chunks:
        raise RuntimeError("沒有產生任何模擬 chunk，請檢查 lookback/step 設定。")

    return pd.concat(sim_chunks, ignore_index=True), param_log
