"""
stat_process.py  v6
===================
純統計過程模型，完全不使用 agent。

Fix-7 — 正確的偏態保留方案
--------------------------------------
問題分析：
  skewnorm(a>0) 的 mean offset = a / sqrt(1+a²) * sqrt(2/π)
  這個 offset 是「右尾多」的數值表現，必須在最後 rescale 前存在。

  失敗的嘗試：
  - v3/Fix-4：兩次 demean → skew=−0.68（小步改善）
  - v4/Fix-5：AR(1) 後只除 std → skew=−0.68 （AR(1) 前的第一次 demean 仍在）
  - v5/Fix-6：Step-1 也只除 std → std 爆炸 0.138，skew=−3
             原因：z[0] 將隨機 offset 帶進 AR(1)，AR(1) 漏抖，mem 尺度就崩

  正確方案（本版）：
  1. 从 skewnorm 取樣後，記錄 skew_mean_offset = a/sqrt(1+a²)*sqrt(2/pi)
  2. full standardize：(z - mean) / std → AR(1) 在正規分佈上適攮標準化
  3. AR(1) 完成後，(mem - mean) / std （AR(1) 本身不改變偏態）
  4. 將 skew_mean_offset 加回去：z_final = z_ar1 + skew_mean_offset
  5. rescale：log_rets = z_final * ret_std + ret_mu
     此時 z_final 的 mean = skew_mean_offset，右尾偏移正確匙入

v1-v6 修正歷程
--------------
  Fix-1 : Student-t df 掃描 log-likelihood
  Fix-2 : 改用 skewnorm 擬合
  Fix-3 : wick_lambda 使用 rolling ATR (Wilder 14)
  Fix-4 : AR(1) 正規化使用 skewnorm 實際 std
  Fix-5 : AR(1) 後只除 std，不 demean
  Fix-6 : 失敗—Step-1 只除 std 導致 AR(1) 漏扖、std 爆炸
  Fix-7 : 正確方案—保存 skew_mean_offset，在 AR(1) 後手動加回
"""

from __future__ import annotations

import warnings
from typing import TypedDict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from sim.metrics import hurst_exponent


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

class StatParams(TypedDict):
    ret_mu:       float   # sample mean of log returns
    ret_std:      float   # sample std  of log returns
    ret_skew_a:   float   # skewnorm shape (alpha)
    ret_df:       float   # Student-t df  (tail thickness, diagnostic only)
    hurst_target: float   # Hurst exponent [0.3, 0.8]
    wick_lambda:  float   # Exponential scale for wick, in units of ATR
    atr_mean:     float   # Mean ATR in absolute price (for wick generation)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_returns(closes: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.maximum(closes.astype(float), 1e-10)))


def _wilder_atr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(hi)
    tr = np.empty(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
    atr = np.empty(n)
    atr[:period] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def _fit_df_scan(log_rets: np.ndarray) -> float:
    """Fix-1: bounded 1-D scan for best Student-t df in [2.1, 30]."""
    mu    = float(np.mean(log_rets))
    sigma = float(np.std(log_rets, ddof=1))
    def neg_ll(df):
        return -np.sum(stats.t.logpdf(log_rets, df=df, loc=mu, scale=sigma))
    return float(minimize_scalar(neg_ll, bounds=(2.1, 30.0), method="bounded").x)


def _fit_skewnorm(log_rets: np.ndarray) -> tuple[float, float, float]:
    """Fix-2: MLE fit skewnorm, return (skew_a, loc, scale)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            a, loc, scale = stats.skewnorm.fit(log_rets)
            a     = float(np.clip(a, -10.0, 10.0))
            scale = float(max(scale, 1e-6))
            loc   = float(loc)
        except Exception:
            a, loc, scale = 0.0, float(np.mean(log_rets)), float(np.std(log_rets))
    return a, loc, scale


def _fit_wick_lambda(df_ohlc: pd.DataFrame) -> tuple[float, float]:
    """Fix-3: normalize wick by Wilder ATR, return (wick_lambda, atr_mean)."""
    hi = df_ohlc["High"].values.astype(float)
    lo = df_ohlc["Low"].values.astype(float)
    op = df_ohlc["Open"].values.astype(float)
    cl = df_ohlc["Close"].values.astype(float)
    body_hi    = np.maximum(op, cl)
    body_lo    = np.minimum(op, cl)
    upper_wick = np.maximum(hi - body_hi, 0.0)
    lower_wick = np.maximum(body_lo - lo, 0.0)
    atr        = np.maximum(_wilder_atr(hi, lo, cl, period=14), 1e-10)
    atr_mean   = float(np.mean(atr))
    wick_ratio = np.concatenate([upper_wick / atr, lower_wick / atr])
    wick_ratio = wick_ratio[wick_ratio > 0]
    if len(wick_ratio) < 10:
        return 0.3, atr_mean
    return float(np.mean(wick_ratio)), atr_mean


def _skewnorm_mean_offset(a: float) -> float:
    """Theoretical mean of skewnorm(a, loc=0, scale=1) = a/sqrt(1+a^2) * sqrt(2/pi)."""
    return float(a / np.sqrt(1.0 + a * a) * np.sqrt(2.0 / np.pi))


# ---------------------------------------------------------------------------
# 1. FIT
# ---------------------------------------------------------------------------

def fit(df_history: pd.DataFrame) -> StatParams:
    closes   = df_history["Close"].values
    log_rets = _log_returns(closes)
    if len(log_rets) < 5:
        raise ValueError(f"lookback too short ({len(log_rets)} bars), need >= 5.")

    ret_mu  = float(np.mean(log_rets))
    ret_std = float(np.std(log_rets, ddof=1))

    df_t               = _fit_df_scan(log_rets)
    skew_a, _, _       = _fit_skewnorm(log_rets)
    h                  = hurst_exponent(log_rets)
    wick_lam, atr_mean = _fit_wick_lambda(df_history)

    return StatParams(
        ret_mu       = ret_mu,
        ret_std      = ret_std,
        ret_skew_a   = skew_a,
        ret_df       = df_t,
        hurst_target = float(np.clip(h, 0.3, 0.8)),
        wick_lambda  = wick_lam,
        atr_mean     = atr_mean,
    )


# ---------------------------------------------------------------------------
# 2. GENERATE
# ---------------------------------------------------------------------------

def _ar1_hurst_rho(h: float) -> float:
    return float(np.clip(2 ** (2 * h - 1) - 1, -0.95, 0.95))


def generate(
    params:      StatParams,
    n_bars:      int,
    start_price: float = 100.0,
    seed:        int | None = None,
) -> pd.DataFrame:
    """
    Fix-7 generation pipeline (correct skew-preserving design)
    -----------------------------------------------------------
    1. Draw z ~ skewnorm(a, loc=0, scale=1)
       Compute and save skew_offset = theoretical mean = a/sqrt(1+a^2)*sqrt(2/pi)
       Full-standardize z: (z - mean(z)) / std(z)  -> AR(1) gets zero-mean input

    2. AR(1) in unit space
       Full-standardize output: (mem - mean(mem)) / std(mem)
       AR(1) preserves autocorrelation structure, not skewness -- that's fine.

    3. Re-inject skew: z_final = z_ar1 + skew_offset
       Now z_final has mean = skew_offset (positive for right-skew, negative for left)
       and std ~ 1 (skew_offset is small, O(0.1-0.5) for typical a values).

    4. Rescale: log_rets = z_final * ret_std + ret_mu
       The skew direction is carried by skew_offset in z_final.
    """
    rng = np.random.default_rng(seed)

    ret_mu   = params["ret_mu"]
    ret_std  = params["ret_std"]
    skew_a   = params["ret_skew_a"]
    hurst    = params["hurst_target"]
    wick_lam = params["wick_lambda"]
    atr_mean = params["atr_mean"]

    # Step 1: draw and save skew offset, then full-standardize
    skew_offset = _skewnorm_mean_offset(skew_a)
    z = stats.skewnorm.rvs(a=skew_a, loc=0, scale=1, size=n_bars, random_state=rng)
    z_std = np.std(z)
    if z_std > 1e-10:
        z = (z - np.mean(z)) / z_std   # zero-mean unit-std for stable AR(1)

    # Step 2: AR(1) in standardized space, full-standardize output
    rho = _ar1_hurst_rho(hurst)
    if abs(rho) > 0.01:
        mem = np.empty(n_bars)
        mem[0] = z[0]
        innov_scale = np.sqrt(1.0 - rho ** 2)
        for i in range(1, n_bars):
            mem[i] = rho * mem[i-1] + innov_scale * z[i]
        m_std = np.std(mem)
        if m_std > 1e-10:
            mem = (mem - np.mean(mem)) / m_std
        z = mem

    # Step 3: re-inject theoretical skew mean offset
    z = z + skew_offset

    # Step 4: rescale to real sample statistics
    log_rets = z * ret_std + ret_mu

    # Rebuild OHLC
    opens  = np.empty(n_bars)
    closes = np.empty(n_bars)
    opens[0] = start_price
    for i in range(n_bars):
        if i > 0:
            opens[i] = closes[i - 1]
        closes[i] = opens[i] * np.exp(log_rets[i])

    atr_proxy   = np.full(n_bars, atr_mean)
    upper_wicks = rng.exponential(scale=wick_lam * atr_proxy)
    lower_wicks = rng.exponential(scale=wick_lam * atr_proxy)
    body_hi     = np.maximum(opens, closes)
    body_lo     = np.minimum(opens, closes)

    volumes = rng.lognormal(mean=15.0, sigma=0.5, size=n_bars).astype(int)

    return pd.DataFrame({
        "Open":   opens,
        "High":   body_hi + upper_wicks,
        "Low":    body_lo - lower_wicks,
        "Close":  closes,
        "Volume": volumes,
    })


# ---------------------------------------------------------------------------
# 3. ROLLING FIT -> GENERATE
# ---------------------------------------------------------------------------

def rolling_fit_generate(
    df_real:   pd.DataFrame,
    lookback:  int = 60,
    step:      int = 20,
    n_forward: int | None = None,
    seed:      int = 42,
    verbose:   bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
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

        param_log.append({
            "window":   window_idx + 1,
            "fit_bars": [fit_start, fit_end],
            "fwd_bars": [pos, fwd_end],
            **{k: params[k] for k in params},
            "loss":     round(loss, 4),
        })

        if verbose:
            print(
                f"[stat] window {window_idx+1:>3}  "
                f"fit=[{fit_start}:{fit_end}]  fwd=[{pos}:{fwd_end}]  "
                f"df={params['ret_df']:.2f}  "
                f"skew_a={params['ret_skew_a']:+.3f}  "
                f"std={params['ret_std']:.4f}  "
                f"hurst={params['hurst_target']:.3f}  "
                f"wick={params['wick_lambda']:.3f}  "
                f"loss={loss:.4f}"
            )

        pos += step
        window_idx += 1

    if not sim_chunks:
        raise RuntimeError("No chunks generated -- check lookback/step settings.")

    return pd.concat(sim_chunks, ignore_index=True), param_log
