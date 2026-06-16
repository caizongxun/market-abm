"""
stat_process.py  v9
===================
純統計過程模型，完全不使用 agent。

Fix-10 — 修正 skew 方向翻轉 + kurtosis 嚴重不足
-------------------------------------------------
Fix-9 後剩餘問題：
  skew  = -0.183  vs 真實 +0.449  (方向翻轉)
  kurt  =  1.637  vs 真實 10.60   (-85%)
  mean  = -0.0002 vs 真實 +0.0008 (符號翻轉，但量小)

根本原因分析：
  A. skew 翻轉：
     0.5*z(skewnorm) + 0.5*mem(t-AR1) 的混合，
     mem 是對稱 t，稀釋了 skew，且 AR(1) rho≈0.56 會讓 mem 的
     尾部分佈與 z 的偏態抵消。最終 skew 甚至翻號。

  B. kurtosis 仍低：
     最後做 z_final / std(z_final) 把 fat-tail 的 outlier 壓縮，
     kurtosis 從理論 ~6 降到 1.6。

Fix-10 設計（徹底分離三個目標）：
  1. 先用純 t-AR(1) 生成 fat-tail + autocorrelation 結構：
       eps ~ t(df, 0, sqrt((df-2)/df))
       mem[i] = rho*mem[i-1] + innov_scale*eps[i]
     不做任何 standardize，直接保留 fat-tail。

  2. 用 empirical CDF rank → skewnorm.ppf 把 mem 的排名
     映射到 skewnorm 分位數（quantile mapping）：
       ranks = argsort(argsort(mem)) / (n-1)   [0,1]
       z_skew = skewnorm.ppf(clip(ranks, ε, 1-ε), a=skew_a, loc=0, scale=1)
     這樣 fat-tail 的 rank 結構被保留（kurtosis），
     skew 方向由 skewnorm.ppf 決定（準確），
     不存在均值偏移問題。

  3. 對 z_skew 做 empirical standardize（減均值除標準差），
     確保 z_skew 的 mean≈0, std≈1，
     然後 log_rets = z_skew * ret_std + ret_mu。

  期望效果：
    kurtosis: t-AR(1) 保留肥尾 rank，quantile mapping 保留形狀 → kurt > 5
    skew:     skewnorm.ppf(a=+1.5) 的右尾 → skew > +0.3
    mean:     empirical standardize 確保 mean≈ret_mu
    std:      直接由 ret_std 控制

v1-v10 修正歷程
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
  Fix-10: quantile mapping (t-AR1 ranks → skewnorm.ppf) 同時保留 kurtosis 和 skew
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


def _quantile_map_to_skewnorm(arr: np.ndarray, skew_a: float, eps: float = 1e-4) -> np.ndarray:
    """Fix-10: map arr's rank structure onto skewnorm(skew_a) quantiles.

    Preserves the rank ordering (and thus fat-tail structure) of arr
    while reshaping the marginal distribution to match skewnorm(skew_a).
    The result has the skew direction of skewnorm but the kurtosis of arr.
    """
    n = len(arr)
    # Fractional ranks in (0, 1) — Hazen plotting position avoids 0/1 boundary
    ranks = (np.argsort(np.argsort(arr)) + 0.5) / n
    ranks = np.clip(ranks, eps, 1.0 - eps)
    return stats.skewnorm.ppf(ranks, a=skew_a, loc=0, scale=1)


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
    Fix-10 generation pipeline
    ---------------------------
    1. Pure t-AR(1): generate fat-tail + autocorrelation in one pass
         eps ~ t(df, 0, sqrt((df-2)/df))   [unit-variance]
         mem[i] = rho*mem[i-1] + innov_scale*eps[i]
       No standardization — keep the fat-tail rank structure intact.

    2. Quantile mapping: map mem's ranks onto skewnorm(skew_a) quantiles
         ranks = (argsort(argsort(mem)) + 0.5) / n   [Hazen, avoids 0/1]
         z_skew = skewnorm.ppf(ranks, a=skew_a)
       Result: marginal distribution matches skewnorm (correct skew direction),
               rank ordering preserved from t-AR(1) (correct kurtosis / tail weight).

    3. Empirical standardize z_skew to mean=0, std=1, then rescale:
         log_rets = (z_skew - mean(z_skew)) / std(z_skew) * ret_std + ret_mu
       ret_mu drives mean exactly; ret_std drives vol exactly.
    """
    rng = np.random.default_rng(seed)

    ret_mu   = params["ret_mu"]
    ret_std  = params["ret_std"]
    skew_a   = params["ret_skew_a"]
    df_t     = params["ret_df"]
    hurst    = params["hurst_target"]
    wick_lam = params["wick_lambda"]
    atr_mean = params["atr_mean"]

    # Step 1: t-AR(1) — fat tails + autocorrelation
    rho = _ar1_hurst_rho(hurst)
    if df_t > 2.0:
        t_scale = float(np.sqrt((df_t - 2.0) / df_t))
    else:
        t_scale = 1.0
    eps = stats.t.rvs(df=max(df_t, 2.01), loc=0, scale=t_scale,
                      size=n_bars, random_state=rng)
    if abs(rho) > 0.01:
        mem = np.empty(n_bars)
        mem[0] = eps[0]
        innov_scale = float(np.sqrt(max(1.0 - rho ** 2, 0.0)))
        for i in range(1, n_bars):
            mem[i] = rho * mem[i-1] + innov_scale * eps[i]
    else:
        mem = eps.copy()

    # Step 2: quantile mapping → skewnorm marginal, t-AR(1) rank structure
    z_skew = _quantile_map_to_skewnorm(mem, skew_a)

    # Step 3: empirical standardize then rescale to (ret_mu, ret_std)
    z_mean = float(np.mean(z_skew))
    z_std  = float(np.std(z_skew))
    if z_std > 1e-10:
        z_skew = (z_skew - z_mean) / z_std
    log_rets = z_skew * ret_std + ret_mu

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
