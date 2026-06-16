"""
stat_process.py  v13
====================
純統計過程模型，完全不使用 agent。

Fix-13 — 修正 Fix-12 的兩個 bug
---------------------------------------
Fix-12 結果：
  skew   = +0.499  ✅ (真實 +0.449)
  mean   = +0.00371 ❌ (真實 +0.00077, 偏差 4.8x)
  kurtosis = 2.50  ❌ (真實 10.6)

Bug-1 mean 偏移：
  linspace 從超取樣 sorted pool 承接等距分位點。
  skewnorm(a=+1.5) 的尾部正偏尾比負偏尾長，
  linspace 選起麼点會包含更多正偏尾的樣本 =>均値偏移。
  修正：直接用 skewnorm.rvs(size=n_bars) 隨機取樣，
         不用超取樣 + linspace。

Bug-2 kurtosis 不足：
  scipy 的 skewnorm(a) excess kurtosis 公式:
    kurt_excess = (8-3π)δ⁴ / (1-(2/π)δ²)²，對 a=1 約 0.98
  遠遠不夠 10.6。fat tail 必須來自 Student-t。
  但 Fix-9~11 的線性混合會把 skew 拉向 0。

Fix-13 正確方案：
  讓 skew 來自 skewnorm， fat tail 來自 t，透過 quantile-blend 合併。

  Step A: 建立混合尾部分佈的 marginal:
    1. 取 n_large=5000 個 skewnorm(a, 0, 1) 樣本
    2. 取 n_large=5000 個 t(df, 0, 1) 樣本
    3. 按 quantile 混合 (mix_t=0.5):
       對每個分位點 p ∈ (0,1):
         q_sn(p) = skewnorm.ppf(p, a)
         q_t(p)  = t.ppf(p, df)
         q_mix(p) = (1-mix_t)*q_sn(p) + mix_t*q_t(p)
       → q_mix 同時有 skewnorm 的 skew 形狀和 t 的 fat tail
       → 不是線性混合分佈，而是 quantile 層面合併，skew 方向不會被消除

  Step B: 從 q_mix 直接 rvs n_bars 個樣本：
    u_iid ~ Uniform(0,1), size=n_bars
    samples = q_mix(u_iid)  ← 模擬直接從混合分佈取樣

  Step C: Gaussian Copula AR(1) rank-remap (同 Fix-12 Step 2-3):
    samples_sorted = sort(samples)
    u ~ AR(1) Gaussian => rank_of_u
    z_final = samples_sorted[rank_of_u]
    → 邊際 = q_mix 分佈（skew+fat-tail）
    → 相關 = AR(1) 結構

  Step D: Rescale:
    log_rets = z_final * ret_std + ret_mu

  選擇 mix_t=0.5 的依據：
    t(df=8) 的 excess kurtosis = 6/(df-4) = 1.5
    skewnorm(a=1) 的 excess kurtosis ≈ 0.98
    重要： quantile 層面合併的 kurtosis 鄉尾部 q_t 控制，
    mix_t=0.5 後尾部行為接近 t，kurtosis >> skewnorm 單獨。
    AAPL 的真實 kurtosis=10.6 需要 df≈7.5 (excess≈1/(7.5-4)*6=2)
    混合後約 3~5 excess，進步可以用 mix_t=0.7 提高。

v1-v13 修正歷程
--------------
  Fix-1 : Student-t df 掃描 log-likelihood
  Fix-2 : 改用 skewnorm 擬合
  Fix-3 : wick_lambda 使用 rolling ATR (Wilder 14)
  Fix-4~8: AR(1) 正規化、mean offset、rolling anchor
  Fix-9~11: 失敗—線性 t-blend 消除 skew
  Fix-12: Gaussian Copula + skewnorm 邊際 => skew 修復但 kurtosis≈2.5
  Fix-13: quantile-blend (skewnorm + t) 尾部，Gaussian Copula rank-remap
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


def _build_quantile_blend_ppf(
    skew_a: float,
    df_t: float,
    mix_t: float = 0.5,
    n_grid: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a quantile-blended PPF grid from skewnorm + t.
    q_mix(p) = (1-mix_t)*skewnorm.ppf(p,a) + mix_t*t.ppf(p,df)

    Returns (p_grid, q_grid) for interpolation.
    This is NOT linear mixture of distributions.
    It's quantile-level blending: each quantile p gets a value
    that is a weighted average of what skewnorm and t would assign.
    Preserves skew direction (from skewnorm) and fat tails (from t).
    """
    eps = 1e-4
    p_grid = np.linspace(eps, 1.0 - eps, n_grid)
    q_sn   = stats.skewnorm.ppf(p_grid, a=skew_a, loc=0, scale=1)
    q_t    = stats.t.ppf(p_grid, df=max(df_t, 2.01), loc=0, scale=1)
    q_mix  = (1.0 - mix_t) * q_sn + mix_t * q_t
    return p_grid, q_mix


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
# 2. GENERATE  (Fix-13: quantile-blend marginal + Gaussian Copula AR(1))
# ---------------------------------------------------------------------------

def _ar1_hurst_rho(h: float) -> float:
    return float(np.clip(2 ** (2 * h - 1) - 1, -0.95, 0.95))


# mix_t controls fat-tail intensity:
#   0.0 = pure skewnorm  (kurtosis ~1)
#   0.5 = balanced       (kurtosis ~3-4)
#   0.7 = t-heavy        (kurtosis ~6-8)
# AAPL excess kurtosis ~10.6 => use 0.7
MIX_T_DEFAULT = 0.7


def generate(
    params:      StatParams,
    n_bars:      int,
    start_price: float = 100.0,
    seed:        int | None = None,
    mix_t:       float = MIX_T_DEFAULT,
) -> pd.DataFrame:
    """
    Fix-13: Quantile-blend marginal (skewnorm+t) + Gaussian Copula AR(1)
    ---------------------------------------------------------------------
    Marginal distribution = quantile blend of skewnorm and t:
      q_mix(p) = (1-mix_t)*skewnorm.ppf(p,a) + mix_t*t.ppf(p,df)

    This is NOT a linear mixture of densities (which would destroy skew).
    It is a blend at the quantile level:
      - Left tail (p<0.5): q_t >> q_sn in absolute value  => fat left tail
      - Right tail (p>0.5): q_t and q_sn blend, skewnorm shifts right => skew preserved
      - Result: asymmetric fat tails with skewnorm's skew direction intact

    Autocorrelation is added via Gaussian Copula rank-remap (Fix-12 method).

    Steps:
      A. Build q_mix PPF grid from (skew_a, df_t, mix_t)
      B. Sample n_bars uniform values -> pass through q_mix -> i.i.d. samples
         with target marginal
      C. Generate Gaussian AR(1) -> get rank ordering
      D. Rank-remap: assign sorted samples to AR(1) ranks
      E. Rescale: log_rets = z_final * ret_std + ret_mu
    """
    rng = np.random.default_rng(seed)

    ret_mu   = params["ret_mu"]
    ret_std  = params["ret_std"]
    skew_a   = params["ret_skew_a"]
    df_t     = params["ret_df"]
    hurst    = params["hurst_target"]
    wick_lam = params["wick_lambda"]
    atr_mean = params["atr_mean"]

    # Step A: build quantile-blend PPF grid
    p_grid, q_grid = _build_quantile_blend_ppf(skew_a, df_t, mix_t=mix_t)

    # Step B: i.i.d. samples from blended marginal
    # Generate uniform(0,1) -> interpolate through q_mix
    u_iid = rng.uniform(0.0, 1.0, size=n_bars)
    samples = np.interp(u_iid, p_grid, q_grid)
    samples_sorted = np.sort(samples)

    # Step C: Gaussian AR(1) for rank ordering
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

    # Step D: rank-remap (Gaussian Copula)
    rank_of_ar1 = np.argsort(np.argsort(u_ar1))  # 0..n_bars-1
    z_final = samples_sorted[rank_of_ar1]

    # Step E: rescale
    # z_final has mean = E[q_mix(U)] where U~Uniform(0,1)
    # = integral of q_mix dp  = mean of the blended distribution
    # For skewnorm(a,0,1): mean = delta*sqrt(2/pi), not 0.
    # We need to subtract the theoretical mean of the blended distribution
    # and then scale, so that log_rets has exactly ret_mu and ret_std.
    z_mean = float(np.mean(q_grid))   # E[q_mix] = integral over uniform p
    z_std  = float(np.std(q_grid))    # std of q_mix over uniform p
    if z_std > 1e-10:
        z_norm = (z_final - z_mean) / z_std  # mean=0, std=1, skew/kurt preserved
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
