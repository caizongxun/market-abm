"""
stat_process.py  v14
====================
純統計過程模型，完全不使用 agent。

Fix-14 — 修正 skew 符號翻轉問題
---------------------------------------
Fix-13 症狀：
  kurtosis = 4.78 ✅(進步)
  skew     = -0.82 ❌ (真實 +0.45，翻轉且幅度大)
  mean     = +0.00090 ✅

Fix-13 的失敗原因：
  q_mix(p) = (1-mix_t)*skewnorm.ppf(p,a) + mix_t*t.ppf(p,df)
  當 skew_a < 0 的 window（共 18 個）：
    - skewnorm.ppf(p, a=-2) 是左偏的
    - t.ppf(p, df) 是對稱的， mix 後對稱成分導致
      對稱的尾部放大，結合左偏 skewnorm => 左尾更大
    - 18 個負 window 的導致對象決定了整體 skew 為負

  根本問題：
    t.ppf 的對稱放大與 skewnorm 的方向交互 =>
    負偏 window 的對稱尾部放大效果大於正偏 window
    => 整體 skew 擅向負

Fix-14 正確方案：尾部放大器（Tail Amplifier）
  目標：增加 fat-tail 但不改變 skew 方向。
  方法：對 skewnorm.ppf 的尾部分位點把尾部拉長，
         中間分位點保持不變。

  Tail Amplifier 定義：
    t_q(p)  = |t.ppf(p, df)|         ← t 的尾部大小
    n_q(p)  = |norm.ppf(p)|          ← Normal 的尾部大小
    amp(p)  = t_q(p) / (n_q(p) + ε)  ← 尾部放大倍數（>1 在尾部）

    q_mix(p) = skewnorm.ppf(p, a) * amp(p)

    特性：
      - amp(p) > 1 在 p < 0.1 和 p > 0.9  => 尾部被拉長
      - amp(p) ≈ 1 在 p ≈ 0.5           => 中間不變
      - amp(p) 是關於 p=0.5 對稱的     => 不改變左右尾部的相對大小
      - 因此 skewnorm 的算術 skew 符號被完全保留

    kurtosis 來源：
      amp(p) 在 p→0 和 p→1 時的增長幾乎跟 t(df) 的尾部行為一致
      => 整體 kurtosis 接近 t(df) 的水準

  Rescale：
    z_mean = E[q_mix] = E[skewnorm(a)*amp]
    z_std  = std[q_mix]
    以 p_grid 上的 q_grid 計算 (uniform 積分)
    z_norm = (z - z_mean) / z_std
    log_rets = z_norm * ret_std + ret_mu

v1-v14 修正歷程
--------------
  Fix-1~3 : df 掃描、skewnorm、rolling ATR wick
  Fix-4~8 : AR(1) 正規化、mean offset、rolling anchor
  Fix-9~11: 失敗—線性 t-blend 消除 skew
  Fix-12  : Gaussian Copula + skewnorm 邊際 => skew 修復但 kurtosis≈2.5
  Fix-13  : quantile-blend (skewnorm+t) => kurtosis 4.78 但 skew 翻轉 -0.82
  Fix-14  : Tail Amplifier = t/normal 尾部比值 => kurtosis↑ + skew 符號保留
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
    Build a tail-amplified PPF:
      q_mix(p) = skewnorm.ppf(p, a) * amp(p)
    where:
      amp(p) = |t.ppf(p, df)| / (|norm.ppf(p)| + eps)

    amp(p) > 1 at tails (p near 0 or 1), amp(p) ≈ 1 at center.
    Symmetric around p=0.5, so skew sign is fully preserved.
    Fat-tail magnitude ≈ t(df) tails.
    """
    eps = 1e-4
    p = np.linspace(eps, 1.0 - eps, n_grid)

    q_sn   = stats.skewnorm.ppf(p, a=skew_a, loc=0, scale=1)
    q_t    = stats.t.ppf(p, df=max(df_t, 2.01), loc=0, scale=1)
    q_norm = stats.norm.ppf(p, loc=0, scale=1)

    # Tail amplifier: how much fatter t is vs normal at each quantile
    amp = np.abs(q_t) / (np.abs(q_norm) + 1e-8)
    # Clip amp to avoid extreme blowup at very deep tails
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
        hurst_target = float(np.clip(h, 0.3, 0.8)),
        wick_lambda  = wick_lam,
        atr_mean     = atr_mean,
    )


# ---------------------------------------------------------------------------
# 2. GENERATE  (Fix-14: Tail Amplifier + Gaussian Copula)
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
    Fix-14: Tail Amplifier marginal + Gaussian Copula AR(1)
    --------------------------------------------------------
    Marginal distribution:
      q_mix(p) = skewnorm.ppf(p, a) * amp(p)
      amp(p) = |t.ppf(p,df)| / |norm.ppf(p)|   (symmetric around p=0.5)

    Properties:
      - amp(p=0.5) = 1  => center unchanged
      - amp(p->0 or 1) > 1  => tails amplified like Student-t
      - amp symmetric => skew sign of skewnorm is preserved exactly
      - kurtosis elevated toward t(df) level

    Autocorrelation: Gaussian Copula rank-remap (same as Fix-12).

    Rescale: subtract E[q_mix], divide by std[q_mix] from p_grid
    (theoretical moments, not empirical -> no tail compression).
    """
    rng = np.random.default_rng(seed)

    ret_mu   = params["ret_mu"]
    ret_std  = params["ret_std"]
    skew_a   = params["ret_skew_a"]
    df_t     = params["ret_df"]
    hurst    = params["hurst_target"]
    wick_lam = params["wick_lambda"]
    atr_mean = params["atr_mean"]

    # Build tail-amplified PPF grid
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

    # Rescale using theoretical moments (no empirical std compression)
    if z_std > 1e-10:
        z_norm = (z_final - z_mean) / z_std
    else:
        z_norm = z_final - z_mean
    log_rets = z_norm * ret_std + ret_mu

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
