"""
stat_process.py  v15
====================
純統計過程模型，完全不使用 agent。

Fix-15 — 三項修正
---------------------------------------
Fix-14 遺留問題：
  skew     = -0.525 ❌ (真實 +0.449，仍然翻轉)
  kurtosis =  3.841 ❌ (真實 10.599，差距 64%)
  hurst    =  0.811 ❌ (真實 0.693，偏高)

Fix-15 修正方案：

  [1] 中心遮罩 amp（修復 skew 翻轉）
      舊問題：|t.ppf(p)|/|norm.ppf(p)| 在 p→0.5 時
               norm.ppf(0.5)=0，分母趨近 0，
               造成中心段數值爆炸後被 clip 打到非 1 的值，
               使 amp 對 p=0.5 不精確對稱 => skew 被扭曲。
      新方案：center_mask = |p-0.5| < 0.35（中心 70% 精確設 amp=1）
               tail_mask = ~center_mask（外側 30% 才做 t/norm 放大）
               => 中心不受數值問題影響，skew 符號完全保留

  [2] Variance-Mixture tail booster（修復 kurtosis）
      舊問題：t(df~10) 的 excess kurtosis 上限 = 6/(df-4) ≈ 1.5，
               但 AAPL 真實 excess kurtosis ≈ 10.6，差距太大。
      新方案：在 rescale 後對 log_rets 乘以 chi2 縮放因子：
               nu_boost = max(df * 0.5, 3.0)  ← 比擬合的 df 更肥
               chi2_scale = sqrt(nu_boost / chi2(nu_boost))
               => 這是 Student-t 的 variance-mixture 構造，
                  有效提升 kurtosis 而不改變 skew 符號
               clip(0.5, 3.0) 防止極端值爆炸

  [3] hurst_target clip 上限 0.8 → 0.72（修復 autocorrelation）
      舊問題：h=0.8 => AR(1) rho = 2^(2*0.8-1)-1 = 0.56，
               自相關過強 => 模擬 hurst=0.811 vs 真實 0.693。
      新方案：clip(0.3, 0.72) => 最大 rho = 2^(2*0.72-1)-1 = 0.33

v1-v15 修正歷程
--------------
  Fix-1~3 : df 掃描、skewnorm、rolling ATR wick
  Fix-4~8 : AR(1) 正規化、mean offset、rolling anchor
  Fix-9~11: 失敗—線性 t-blend 消除 skew
  Fix-12  : Gaussian Copula + skewnorm 邊際 => skew 修復但 kurtosis≈2.5
  Fix-13  : quantile-blend (skewnorm+t) => kurtosis 4.78 但 skew 翻轉 -0.82
  Fix-14  : Tail Amplifier = t/normal 尾部比值 => kurtosis↑ 但 skew 仍翻轉 -0.525
  Fix-15  : center-masked amp + variance-mixture booster + hurst clip 0.72
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
    ret_mu:       float
    ret_std:      float
    ret_skew_a:   float
    ret_df:       float
    hurst_target: float
    wick_lambda:  float
    atr_mean:     float


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
    mu    = float(np.mean(log_rets))
    sigma = float(np.std(log_rets, ddof=1))
    def neg_ll(df):
        return -np.sum(stats.t.logpdf(log_rets, df=df, loc=mu, scale=sigma))
    return float(minimize_scalar(neg_ll, bounds=(2.1, 30.0), method="bounded").x)


def _fit_skewnorm(log_rets: np.ndarray) -> tuple[float, float, float]:
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


def _build_tail_amplified_ppf(
    skew_a: float,
    df_t: float,
    n_grid: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fix-15: Center-masked tail amplifier.

    amp(p) = 1                                     if |p-0.5| < 0.35  (center 70%)
           = |t.ppf(p,df)| / (|norm.ppf(p)| + eps)  if |p-0.5| >= 0.35 (outer tails)

    Center 70% is exactly 1 to avoid numerical issues at p=0.5 where
    norm.ppf(0.5)=0 would blow up the ratio in Fix-14.
    The outer 30% (p < 0.15 or p > 0.85) gets t-like fat tails.
    amp is symmetric around p=0.5, so skew sign of skewnorm is fully preserved.
    """
    eps = 1e-4
    p = np.linspace(eps, 1.0 - eps, n_grid)

    q_sn   = stats.skewnorm.ppf(p, a=skew_a, loc=0, scale=1)
    q_t    = stats.t.ppf(p, df=max(df_t, 2.01), loc=0, scale=1)
    q_norm = stats.norm.ppf(p, loc=0, scale=1)

    # Fix-15: center-masked amp — center 70% is exactly 1
    amp = np.ones_like(p)
    tail_mask = np.abs(p - 0.5) >= 0.35
    amp[tail_mask] = np.abs(q_t[tail_mask]) / (np.abs(q_norm[tail_mask]) + 1e-8)
    amp = np.clip(amp, 0.5, 8.0)

    q_mix = q_sn * amp
    return p, q_mix


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
        # Fix-15: hurst clip upper bound lowered 0.8 -> 0.72
        # h=0.72 => rho = 2^(2*0.72-1)-1 = 0.33  (was 0.56 at h=0.8)
        hurst_target = float(np.clip(h, 0.3, 0.72)),
        wick_lambda  = wick_lam,
        atr_mean     = atr_mean,
    )


# ---------------------------------------------------------------------------
# 2. GENERATE  (Fix-15: center-masked amp + variance-mixture kurtosis booster)
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
    Fix-15: center-masked amp + variance-mixture booster + hurst clip 0.72

    Marginal distribution:
      q_mix(p) = skewnorm.ppf(p, a) * amp(p)
      amp(p) = 1 for |p-0.5| < 0.35  (center 70%, exact)
             = |t.ppf(p,df)| / |norm.ppf(p)|  for tails

    Variance-mixture kurtosis booster:
      nu_boost = max(df * 0.5, 3.0)
      chi2_scale = sqrt(nu_boost / chi2(nu_boost))
      log_rets *= chi2_scale
      => Equivalent to drawing from a heavier-tailed variance-mixture distribution.
         Kurtosis scales inversely with nu_boost (smaller => fatter tails).
         chi2_scale is positive => skew sign is fully preserved.
         Clipped to [0.5, 3.0] to prevent extreme outliers.

    Autocorrelation:
      hurst_target clipped to [0.3, 0.72] => max AR(1) rho = 0.33
    """
    rng = np.random.default_rng(seed)

    ret_mu   = params["ret_mu"]
    ret_std  = params["ret_std"]
    skew_a   = params["ret_skew_a"]
    df_t     = params["ret_df"]
    hurst    = params["hurst_target"]
    wick_lam = params["wick_lambda"]
    atr_mean = params["atr_mean"]

    # Build tail-amplified PPF grid (Fix-15: center-masked)
    p_grid, q_grid = _build_tail_amplified_ppf(skew_a, df_t)

    # Theoretical mean and std from q_grid (uniform measure over p)
    z_mean = float(np.mean(q_grid))
    z_std  = float(np.std(q_grid))

    # i.i.d. samples from blended marginal
    u_iid = rng.uniform(0.0, 1.0, size=n_bars)
    samples = np.interp(u_iid, p_grid, q_grid)
    samples_sorted = np.sort(samples)

    # Gaussian AR(1) copula
    rho = _ar1_hurst_rho(hurst)
    eps = rng.standard_normal(n_bars)
    if abs(rho) > 0.01:
        u_ar1 = np.empty(n_bars)
        u_ar1[0] = eps[0]
        innov_scale = float(np.sqrt(max(1.0 - rho ** 2, 0.0)))
        for i in range(1, n_bars):
            u_ar1[i] = rho * u_ar1[i-1] + innov_scale * eps[i]
    else:
        u_ar1 = eps.copy()

    # Rank remap
    rank_of_ar1 = np.argsort(np.argsort(u_ar1))
    z_final = samples_sorted[rank_of_ar1]

    # Rescale using theoretical moments
    if z_std > 1e-10:
        z_norm = (z_final - z_mean) / z_std
    else:
        z_norm = z_final - z_mean
    log_rets = z_norm * ret_std + ret_mu

    # Fix-15: Variance-mixture tail booster for kurtosis
    # nu_boost << df => chi2_scale has fatter tails than raw t(df)
    nu_boost = float(max(df_t * 0.5, 3.0))
    chi2_samples = rng.chisquare(nu_boost, size=n_bars)
    chi2_scale = np.sqrt(nu_boost / np.maximum(chi2_samples, 1e-8))
    chi2_scale = np.clip(chi2_scale, 0.5, 3.0)
    log_rets = log_rets * chi2_scale

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
    volumes     = rng.lognormal(mean=15.0, sigma=0.5, size=n_bars).astype(int)

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
