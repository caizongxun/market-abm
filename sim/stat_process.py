"""
stat_process.py  v8
===================
純統計過程模型，完全不使用 agent。

Fix-9 — 修正 mean 雙重計算 + 恢復 kurtosis
----------------------------------------------
問題 A：mean=0.00357 vs 真實 0.00077 (+364%)
  原因：ret_mu 已是 sample mean（包含 skew 效果），
        但 Fix-7 又額外加了 skew_offset * ret_std（~0.0106/bar）。
        相當於把 skew 對 mean 的貢獻計算了兩次。

問題 B：kurtosis=1.23 vs 真實 10.6（-88%）
  原因：AR(1) 輸出做 (mem-mean)/std 正規化，把 fat-tail 資訊磨掉了。
        Student-t 的肥尾必須在 AR(1) 的 innovations 裡，而不是事後加。

Fix-9 正確方案：
  1. z ~ skewnorm(a, loc=0, scale=1)
     moment-preserving standardize：
       mu_sn  = a / sqrt(1+a²) * sqrt(2/π)          [理論平均]
       var_sn = 1 - mu_sn²                            [理論方差]
       z_std  = (z - mu_sn) / sqrt(var_sn)
     → mean=0, std=1, 偏態形狀保留（skew≈0.30 for a=1.5）
     → 不再需要事後加 skew_offset，消除雙重計算

  2. AR(1) 使用 Student-t innovations（df 來自 Fix-1 的掃描）：
       eps_t ~ t(df, 0, sqrt((df-2)/df))   [unit-variance t noise]
       mem[i] = rho*mem[i-1] + innov_scale*eps_t[i]
     AR(1) output 不做 standardize，直接保留 fat-tail shape。
     為了防止 scale 漂移，只做 sigma-clamp（不 demean，不 rescale）：
       clip mem to ±5σ，然後 rescale by empirical std

  3. log_rets = z_final * ret_std + ret_mu
     z_final 的 mean≈0（moment-preserving），偏態在分佈形狀中，fat-tail 在 innovations 中。

v1-v9 修正歷程
--------------
  Fix-1 : Student-t df 掃描 log-likelihood
  Fix-2 : 改用 skewnorm 擬合
  Fix-3 : wick_lambda 使用 rolling ATR (Wilder 14)
  Fix-4 : AR(1) 正規化使用 skewnorm 實際 std
  Fix-5 : AR(1) 後只除 std，不 demean
  Fix-6 : 失敗—Step-1 只除 std 導致 AR(1) 漏扖、std 爆炸
  Fix-7 : 保存 skew_mean_offset，在 AR(1) 後手動加回（但造成雙重計算）
  Fix-8 : rolling loop 改用 sim last-close 作為下一 chunk 起始價
  Fix-9 : moment-preserving skewnorm standardize + t innovations in AR(1)
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
    ret_df:       float   # Student-t df  (tail thickness)
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


def _skewnorm_moment_standardize(z: np.ndarray, a: float) -> np.ndarray:
    """Fix-9A: moment-preserving standardization of skewnorm samples.

    Subtracts the theoretical mean and divides by theoretical std,
    giving mean=0, std=1 while PRESERVING the skew shape.
    This avoids re-adding skew_offset later (which caused double-counting).
    """
    mu_sn  = a / np.sqrt(1.0 + a * a) * np.sqrt(2.0 / np.pi)
    var_sn = 1.0 - mu_sn ** 2
    std_sn = float(np.sqrt(max(var_sn, 1e-10)))
    return (z - mu_sn) / std_sn


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
    Fix-9 generation pipeline
    --------------------------
    1. Draw z ~ skewnorm(a, 0, 1)
       moment-preserving standardize: subtract theoretical mean, divide by theoretical std
       → z has mean=0, std=1, skew shape preserved (no skew_offset injection needed)

    2. AR(1) with Student-t(df) innovations to preserve fat tails
       eps ~ t(df) scaled to unit variance: scale = sqrt((df-2)/df)
       mem[i] = rho*mem[i-1] + innov_scale*eps[i]
       Rescale mem by empirical std only (no demean) to keep scale correct
       Blend z (skew shape) with mem (fat-tail AR structure): z_final = 0.5*z + 0.5*mem

    3. log_rets = z_final * ret_std + ret_mu
       ret_mu is sample mean which already encodes skew drift; no extra offset needed
    """
    rng = np.random.default_rng(seed)

    ret_mu   = params["ret_mu"]
    ret_std  = params["ret_std"]
    skew_a   = params["ret_skew_a"]
    df_t     = params["ret_df"]
    hurst    = params["hurst_target"]
    wick_lam = params["wick_lambda"]
    atr_mean = params["atr_mean"]

    # Step 1: skewnorm with moment-preserving standardization (Fix-9A)
    z_raw = stats.skewnorm.rvs(a=skew_a, loc=0, scale=1, size=n_bars, random_state=rng)
    z = _skewnorm_moment_standardize(z_raw, skew_a)   # mean=0, std≈1, skew preserved

    # Step 2: AR(1) with Student-t innovations (Fix-9B)
    rho = _ar1_hurst_rho(hurst)
    if abs(rho) > 0.01 and df_t > 2.0:
        # Unit-variance t innovations
        t_scale = float(np.sqrt((df_t - 2.0) / df_t))
        eps = stats.t.rvs(df=df_t, loc=0, scale=t_scale, size=n_bars, random_state=rng)
        mem = np.empty(n_bars)
        mem[0] = eps[0]
        innov_scale = np.sqrt(max(1.0 - rho ** 2, 0.0))
        for i in range(1, n_bars):
            mem[i] = rho * mem[i-1] + innov_scale * eps[i]
        # Rescale by empirical std (no demean to avoid killing fat-tail signal)
        m_std = float(np.std(mem))
        if m_std > 1e-10:
            mem = mem / m_std
        # Blend skew shape (z) with fat-tail AR structure (mem)
        z_final = 0.5 * z + 0.5 * mem
        # Final unit rescale
        zf_std = float(np.std(z_final))
        if zf_std > 1e-10:
            z_final = z_final / zf_std
    else:
        z_final = z

    # Step 3: rescale to real sample statistics
    log_rets = z_final * ret_std + ret_mu

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
    df_real:             pd.DataFrame,
    lookback:            int = 60,
    step:                int = 20,
    n_forward:           int | None = None,
    seed:                int = 42,
    real_anchor_weight:  float = 0.0,
    verbose:             bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Fix-8: track sim_last_close and use it as the next chunk's start_price.

    real_anchor_weight (0..1):
        0.0  — pure chain: start_price = sim_last_close  (no seam jump, default)
        1.0  — always anchor to real close
        0.x  — geometric blend: exp((1-w)*log(sim) + w*log(real))
    """
    if n_forward is None:
        n_forward = step

    n_total    = len(df_real)
    sim_chunks: list[pd.DataFrame] = []
    param_log:  list[dict]         = []
    window_idx = 0
    pos        = lookback
    sim_last_close: float | None = None

    while pos <= n_total:
        fit_start  = pos - lookback
        fit_end    = pos
        fwd_end    = min(pos + n_forward, n_total)
        actual_fwd = fwd_end - pos
        if actual_fwd <= 0:
            break

        df_window = df_real.iloc[fit_start:fit_end].copy().reset_index(drop=True)
        params    = fit(df_window)

        real_close = float(df_real["Close"].iloc[fit_end - 1])
        if sim_last_close is None or real_anchor_weight >= 1.0:
            start_px = real_close
        elif real_anchor_weight <= 0.0:
            start_px = sim_last_close
        else:
            w = float(real_anchor_weight)
            start_px = float(np.exp(
                (1 - w) * np.log(max(sim_last_close, 1e-10))
                + w * np.log(max(real_close, 1e-10))
            ))

        df_chunk = generate(
            params=params, n_bars=actual_fwd,
            start_price=start_px, seed=seed + window_idx,
        )
        sim_chunks.append(df_chunk)
        sim_last_close = float(df_chunk["Close"].iloc[-1])

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
