"""
stat_process.py  v16
====================
純統計過程模型，完全不使用 agent。

Fix-16 — 三項修正
---------------------------------------
Fix-15 遺留問題：
  std      = 0.02170 ❌ (真實 0.01603，偏高 35%)
  skew     = -0.302  ❌ (真實 +0.449，符號仍錯)
  kurtosis =  9.737  ✅ (from 3.8, 幾乎到位)
  hurst    =  0.776  ❌ (真實 0.693，仍偏高)

[1] std 偏高修正
    原因：chi2_scale = sqrt(nu/chi2(nu))，由於 Jensen 不等式，
           E[sqrt(nu/X)] > sqrt(nu/E[X]) = 1
           => 平均傀向 > 1，導致 std 被放大。
    修正：在乘完 chi2_scale 後重新 rescale：
           log_rets = (log_rets - log_rets.mean()) / log_rets.std() * ret_std + ret_mu
           線性變換，保留 skew 符號和 kurtosis 相對大小。

[2] skew 符號修正
    原因：clip(0.5, 3.0) 是非對稱的——
           chi2_scale 可能小至 0.5（下界切）也可能大至 3.0（上界切）。
           對稱性破壞後，乘法把 skewnorm 兩個尾巴放大或縮小的倍率不同，
           skew 符號因此被扭曲。
    修正：改為對稱 clip(1/c, c)， c=2.5
           正負方向的截斷倍率相同，不破壞乘法對稱性。
    同時把 nu_boost 從 max(df*0.5, 3.0) 提高到 max(df*0.8, 4.0)：
           clip 範圍收窄後對 kurtosis 的貢猫少一點，稍大的 nu_boost 補側。

[3] hurst clip 0.72 → 0.65
    原因：v15 hurst=0.776 vs 真實 0.693，仍偏高。
           chi2_scale 乘法放大了波動群聺性，間接提高 hurst。
    修正：clip(0.3, 0.65)。
           h=0.65 => AR(1) rho = 2^(2*0.65-1)-1 = 0.21

v1-v16 修正歷程
--------------
  Fix-1~3 : df 掃描、skewnorm、rolling ATR wick
  Fix-4~8 : AR(1) 正規化、mean offset、rolling anchor
  Fix-9~11: 失敗—線性 t-blend 消除 skew
  Fix-12  : Gaussian Copula + skewnorm 邊際 => skew 修復但 kurtosis≈2.5
  Fix-13  : quantile-blend (skewnorm+t) => kurtosis 4.78 但 skew 翻轉 -0.82
  Fix-14  : Tail Amplifier = t/normal 尾部比值 => kurtosis↑ 但 skew 仍翻轉
  Fix-15  : center-masked amp + variance-mixture booster + hurst clip 0.72
             => kurtosis 9.74 ✅ 但 std 偏高 35%、skew -0.30、hurst 仍偏
  Fix-16  : symmetric clip + std rescale + nu_boost 0.8x + hurst clip 0.65
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
    Fix-15/16: Center-masked tail amplifier.

    amp(p) = 1                                        if |p-0.5| < 0.35
           = |t.ppf(p,df)| / (|norm.ppf(p)| + eps)   if |p-0.5| >= 0.35

    Center 70% is exactly 1 to avoid the p=0.5 singularity from Fix-14.
    Outer 30% gets t-like fat tails. amp is symmetric around p=0.5,
    so skew sign of skewnorm is preserved.
    """
    eps = 1e-4
    p = np.linspace(eps, 1.0 - eps, n_grid)

    q_sn   = stats.skewnorm.ppf(p, a=skew_a, loc=0, scale=1)
    q_t    = stats.t.ppf(p, df=max(df_t, 2.01), loc=0, scale=1)
    q_norm = stats.norm.ppf(p, loc=0, scale=1)

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
        # Fix-16: hurst clip upper bound 0.72 -> 0.65
        # h=0.65 => AR(1) rho = 2^(2*0.65-1)-1 = 0.21
        hurst_target = float(np.clip(h, 0.3, 0.65)),
        wick_lambda  = wick_lam,
        atr_mean     = atr_mean,
    )


# ---------------------------------------------------------------------------
# 2. GENERATE  (Fix-16)
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
    Fix-16 changes vs Fix-15:

    (A) Symmetric chi2_scale clip: clip(1/2.5, 2.5) instead of clip(0.5, 3.0)
        Reason: asymmetric clip in Fix-15 broke the multiplicative symmetry
        of chi2_scale, causing the skewnorm's skew to be distorted toward negative.
        A symmetric clip preserves the relative magnitude of both tails equally.

    (B) nu_boost = max(df * 0.8, 4.0)  (was max(df * 0.5, 3.0))
        Reason: symmetric clip has a narrower effective range than the old
        asymmetric clip (max ratio 2.5 vs 3.0), slightly reducing kurtosis.
        Compensate by using a slightly larger nu_boost (closer to original df)
        so chi2 samples are still heavy-tailed enough.

    (C) Post-booster std rescale
        Reason: E[sqrt(nu/chi2(nu))] > 1 by Jensen's inequality, so multiplying
        by chi2_scale inflates std. Re-standardise after the multiplication:
          log_rets = (log_rets - mean) / std * ret_std + ret_mu
        This is a linear transform, so skew sign and kurtosis ratio are preserved.

    (D) hurst_target clipped to [0.3, 0.65] => max AR(1) rho = 0.21
    """
    rng = np.random.default_rng(seed)

    ret_mu   = params["ret_mu"]
    ret_std  = params["ret_std"]
    skew_a   = params["ret_skew_a"]
    df_t     = params["ret_df"]
    hurst    = params["hurst_target"]
    wick_lam = params["wick_lambda"]
    atr_mean = params["atr_mean"]

    # Build tail-amplified PPF grid (center-masked, from Fix-15)
    p_grid, q_grid = _build_tail_amplified_ppf(skew_a, df_t)

    # Theoretical mean and std from q_grid
    z_mean = float(np.mean(q_grid))
    z_std  = float(np.std(q_grid))

    # i.i.d. samples from tail-amplified marginal
    u_iid = rng.uniform(0.0, 1.0, size=n_bars)
    samples = np.interp(u_iid, p_grid, q_grid)
    samples_sorted = np.sort(samples)

    # Gaussian AR(1) copula for autocorrelation
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

    # Rank remap: impose autocorrelation structure while preserving marginal
    rank_of_ar1 = np.argsort(np.argsort(u_ar1))
    z_final = samples_sorted[rank_of_ar1]

    # Rescale to target moments
    if z_std > 1e-10:
        z_norm = (z_final - z_mean) / z_std
    else:
        z_norm = z_final - z_mean
    log_rets = z_norm * ret_std + ret_mu

    # Fix-16A+B: Variance-mixture kurtosis booster with SYMMETRIC clip
    # nu_boost = max(df * 0.8, 4.0) — larger than Fix-15's 0.5x to compensate
    # for narrower clip range; still produces heavier tails than raw t(df)
    nu_boost = float(max(df_t * 0.8, 4.0))
    chi2_samples = rng.chisquare(nu_boost, size=n_bars)
    chi2_scale = np.sqrt(nu_boost / np.maximum(chi2_samples, 1e-8))
    # Fix-16A: symmetric clip — same ratio in both directions preserves skew
    _c = 2.5
    chi2_scale = np.clip(chi2_scale, 1.0 / _c, _c)
    log_rets = log_rets * chi2_scale

    # Fix-16C: rescale std back to ret_std (linear => preserves skew sign & kurtosis ratio)
    lr_std = float(np.std(log_rets))
    if lr_std > 1e-10:
        log_rets = (log_rets - float(np.mean(log_rets))) / lr_std * ret_std + ret_mu

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
