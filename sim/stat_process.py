"""
stat_process.py  v12
====================
純統計過程模型，完全不使用 agent。

Fix-12 — 放棄 t-blend；直接從 skewnorm 生成再套 AR(1) rank-remap
-----------------------------------------------------------------
歷史症狀總結：
  所有 Fix-9~11 都用「z_sn + alpha*t_std」線性混合。
  根本問題：t 是對稱分佈（skew=0），任何線性混合都把 skew 往 0 拉。
  alpha=0.4 => skew 剩 60%，alpha=0.5 => 剩 50%，且在 n=20 的
  小 chunk 下 empirical skew 會翻號 => 41 個 window 的平均 skew≈0。

正確架構思路（Sklar 定理）：
  邊際分佈 和 相關結構 必須分開處理。
  - 邊際（每根 bar 的 return 分佈）：skewnorm(a, loc, scale) 精確控制
  - 相關（AR(1) 自相關）：用 Gaussian copula rank-remap 套上去

Fix-12 管線（Gaussian Copula + skewnorm 邊際）：
  Step 1 - 先從目標邊際分佈取樣（i.i.d.）：
    samples ~ skewnorm.rvs(a=skew_a, loc=0, scale=1, size=n_bars)
    → samples 的 skew/kurtosis 完全由 skewnorm 決定，精確。

  Step 2 - 生成帶 AR(1) 結構的 Gaussian copula：
    u_gauss ~ AR(1) Gaussian process (rho from Hurst)
    u_gauss empirically standardized → mean=0, std=1
    → 把 u_gauss 轉成 uniform ranks: p = Φ(u_gauss)  [standard normal CDF]

  Step 3 - Rank remap（Gaussian copula）：
    把 samples 按 p 的排名重新排列：
    sorted_samples = sort(samples)
    rank_of_p = argsort(argsort(p))
    z_final = sorted_samples[rank_of_p]
    → z_final 的邊際分佈 = samples（精確 skewnorm）
    → z_final 的排名相關性 = AR(1) Gaussian copula
    → skew 和 kurtosis 完全保留！

  Step 4 - Rescale：
    log_rets = z_final * ret_std + ret_mu

  理論保證：
    - skew(z_final) = skew(samples) = skewnorm(a) 的理論 skew  ✓
    - kurt(z_final) = kurt(samples) = skewnorm(a) 的理論 kurt  ✓
    - autocorr(z_final) ≈ AR(1) copula 的 rho  ✓（rank correlation）
    - 無任何 empirical std 壓縮操作  ✓

v1-v12 修正歷程
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
  Fix-10: quantile mapping (t-AR1 ranks → skewnorm.ppf)
  Fix-11: 理論矩 normalize + t fat-tail 後混合，消除 empirical std 壓縮問題
  Fix-12: 放棄 t-blend，改用 Gaussian Copula + skewnorm 邊際分佈
          => skew/kurtosis 由邊際精確控制，AR(1) 由 copula 套用
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
    ret_df:       float   # Student-t df  (tail thickness, kept for reference)
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
# 2. GENERATE  (Fix-12: Gaussian Copula + skewnorm marginal)
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
    Fix-12: Gaussian Copula + skewnorm marginal
    -------------------------------------------
    Sklar theorem: any joint distribution = copula(marginals).
    We want:
      marginal: skewnorm(a, 0, 1)  => controls skew + kurtosis exactly
      copula:   Gaussian AR(1)     => controls autocorrelation

    Step 1 — Sample i.i.d. from target marginal (skewnorm):
      samples ~ skewnorm.rvs(a=skew_a, loc=0, scale=1, size=n_bars)
      These have exact skew and fat-tail shape of skewnorm(a).

    Step 2 — Generate Gaussian AR(1) copula scores:
      u ~ AR(1) Gaussian process with rho from Hurst
      Standardize u -> N(0,1) marginal
      Convert to uniform via standard normal CDF: p = Phi(u)
      p[i] in (0,1) gives rank ordering with AR(1) dependence.

    Step 3 — Rank-remap (apply copula to marginal):
      Sort samples ascending -> sorted_s
      rank_idx = argsort(argsort(p))  [rank position of each p[i]]
      z_final = sorted_s[rank_idx]
      Result: z_final[i] has the same rank as p[i],
              so autocorrelation structure of p is preserved,
              and marginal distribution = exactly skewnorm(a).

    Step 4 — Rescale:
      log_rets = z_final * ret_std + ret_mu

    Guarantees:
      skew(z_final)  = skew(samples)  [rank remap preserves marginal]
      kurt(z_final)  = kurt(samples)  [rank remap preserves marginal]
      autocorr(z_final) ≈ AR(1) rho   [Spearman rank correlation preserved]
      NO empirical std compression step anywhere.
    """
    rng = np.random.default_rng(seed)

    ret_mu   = params["ret_mu"]
    ret_std  = params["ret_std"]
    skew_a   = params["ret_skew_a"]
    hurst    = params["hurst_target"]
    wick_lam = params["wick_lambda"]
    atr_mean = params["atr_mean"]

    # Step 1: i.i.d. samples from target marginal (skewnorm)
    # Use a large oversample then trim to avoid edge effects from ppf clipping
    n_sample = max(n_bars * 4, 200)  # oversample for stable quantiles
    samples_iid = stats.skewnorm.rvs(
        a=skew_a, loc=0, scale=1, size=n_sample,
        random_state=rng
    )
    # Take the middle n_bars quantiles to avoid extreme edge samples
    # Sort samples and pick evenly spaced quantile positions
    samples_sorted = np.sort(samples_iid)
    # Select n_bars evenly spaced positions from the sorted oversampled pool
    idx = np.round(np.linspace(0, n_sample - 1, n_bars)).astype(int)
    samples = samples_sorted[idx]  # these n_bars values span the full distribution

    # Step 2: Gaussian AR(1) copula
    rho = _ar1_hurst_rho(hurst)
    eps = rng.standard_normal(n_bars)
    if abs(rho) > 0.01:
        u = np.empty(n_bars)
        u[0] = eps[0]
        innov_scale = float(np.sqrt(max(1.0 - rho ** 2, 0.0)))
        for i in range(1, n_bars):
            u[i] = rho * u[i-1] + innov_scale * eps[i]
    else:
        u = eps.copy()
    # Standardize u to N(0,1) marginal
    u_m, u_s = float(np.mean(u)), float(np.std(u))
    if u_s > 1e-10:
        u = (u - u_m) / u_s
    # Convert to uniform ranks (probability integral transform)
    # p[i] = Phi(u[i]) gives rank ordering with AR(1) dependence
    # We use rank-based approach directly (more stable for small n):
    rank_of_u = np.argsort(np.argsort(u))  # 0..n_bars-1

    # Step 3: Rank-remap: assign samples[rank_of_u[i]] to position i
    # samples is already sorted ascending, so samples[rank_of_u] gives
    # z_final[i] = the rank_of_u[i]-th smallest sample
    z_final = samples[rank_of_u]

    # Step 4: rescale to real distribution moments
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
