"""
stat_process.py  v10
====================
純統計過程模型，完全不使用 agent。

Fix-11 — 根治 skew≈0 和 kurtosis≈1.6
--------------------------------------
Fix-10 症狀：skew=-0.02（真實 +0.45），kurtosis=1.65（真實 10.6）

根本原因（最終確認）：
  empirical standardize step (z_skew - mean) / std 是罪魁禍首：
    1. skewnorm.ppf 已經把 skew 方向正確放入 z_skew，
       但 skewnorm 的 mean ≠ 0（對 a=+1 大約 +0.56）。
       (z_skew - mean(z_skew)) 正確去除了這個 offset。
       但 / std(z_skew) 同時「壓縮」了尾部，把 outlier 往中心壓。
       問題在於 skewnorm 的 std 理論值 ≈ 0.89（a=1），
       但加上 t-tail 後 empirical std 是 ~2.3，
       除以 2.3 把本來 kurtosis=8 的分佈壓到 1.6。

    2. 更麻煩的是：skew 的方向在 small n（n_bars=20）下，
       empirical mean/std 與理論值偏差大，導致 skew 符號不穩定。

Fix-11 正確方案（不做 empirical standardize）：
  1. t-AR(1) 只負責 autocorrelation：
       eps ~ N(0,1)  [對稱，不引入額外 kurtosis 偏差]
       mem[i] = rho*mem[i-1] + innov_scale*eps[i]
       normalize mem to mean=0, std=1  (autocorrelation structure only)

  2. 用 mem 的 rank 做 quantile mapping 到 skewnorm.ppf：
       ranks = (argsort(argsort(mem)) + 0.5) / n
       z_skew = skewnorm.ppf(ranks, a=skew_a, loc=0, scale=1)
     skewnorm.ppf 直接給出正確 skew 方向。

  3. 對 z_skew 做「理論矩」rescale，不用 empirical std（避免壓縮尾部）：
       mu_sn  = a/sqrt(1+a²) * sqrt(2/π)    [skewnorm(a,0,1) 理論均值]
       std_sn = sqrt(1 - mu_sn²)             [skewnorm(a,0,1) 理論標準差]
       z_norm = (z_skew - mu_sn) / std_sn    [理論矩 normalize，不壓縮]
     z_norm 的 mean≈0, std≈1，skew 保留，kurtosis 不被壓縮。

  4. 加入 Student-t fat-tail：
       t_raw ~ t(df, 0, 1)
       t_std = (t_raw - mean(t_raw)) / std(t_raw)   [理論 mean=0, empirical std≈1]
       z_final = alpha * z_norm + (1-alpha) * t_std
     alpha=0.5 讓 skewnorm 的 skew 與 t 的 fat-tail 各貢獻一半。
     t 的 kurtosis(df=8) ≈ 6；混合後仍遠高於 1.6。

  5. log_rets = z_final * ret_std + ret_mu

v1-v11 修正歷程
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


def _skewnorm_theoretical_moments(a: float) -> tuple[float, float]:
    """Return theoretical (mean, std) of skewnorm(a, loc=0, scale=1)."""
    delta  = a / np.sqrt(1.0 + a * a)
    mu_sn  = delta * np.sqrt(2.0 / np.pi)
    var_sn = max(1.0 - mu_sn ** 2, 1e-10)
    return float(mu_sn), float(np.sqrt(var_sn))


def _quantile_map(arr: np.ndarray, skew_a: float, eps: float = 1e-4) -> np.ndarray:
    """Map arr's ranks onto skewnorm(skew_a, 0, 1) quantiles (Hazen position)."""
    n     = len(arr)
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
    Fix-11 generation pipeline
    ---------------------------
    Goal: independently control skew, kurtosis, autocorrelation, mean, vol.

    Step 1 — Autocorrelation skeleton via Gaussian AR(1):
      eps ~ N(0,1);  mem[i] = rho*mem[i-1] + sqrt(1-rho²)*eps[i]
      Empirically standardize mem → mean=0, std=1.
      (Gaussian so it doesn't pollute kurtosis or skew)

    Step 2 — Inject skew via quantile mapping (no post-standardize):
      ranks = (argsort(argsort(mem)) + 0.5) / n     [Hazen]
      z_skew = skewnorm.ppf(ranks, a=skew_a, loc=0, scale=1)
      Theoretical normalize using analytical moments of skewnorm(a,0,1):
        mu_sn, std_sn = _skewnorm_theoretical_moments(a)
        z_sn = (z_skew - mu_sn) / std_sn
      → skew shape preserved; no empirical compression of tails.

    Step 3 — Inject fat tails from Student-t(df):
      t_raw ~ t(df, 0, 1);  empirically standardize → t_std (mean=0, std=1)
      z_final = 0.6 * z_sn + 0.4 * t_std
      Final empirical standardize → mean=0, std=1
      (the blend ensures skew is diluted less than in Fix-9's 50/50)

    Step 4 — Rescale:
      log_rets = z_final * ret_std + ret_mu
    """
    rng = np.random.default_rng(seed)

    ret_mu   = params["ret_mu"]
    ret_std  = params["ret_std"]
    skew_a   = params["ret_skew_a"]
    df_t     = params["ret_df"]
    hurst    = params["hurst_target"]
    wick_lam = params["wick_lambda"]
    atr_mean = params["atr_mean"]

    # Step 1: Gaussian AR(1) for autocorrelation structure
    rho = _ar1_hurst_rho(hurst)
    eps = rng.standard_normal(n_bars)
    if abs(rho) > 0.01:
        mem = np.empty(n_bars)
        mem[0] = eps[0]
        innov_scale = float(np.sqrt(max(1.0 - rho ** 2, 0.0)))
        for i in range(1, n_bars):
            mem[i] = rho * mem[i-1] + innov_scale * eps[i]
    else:
        mem = eps.copy()
    m_std = float(np.std(mem))
    if m_std > 1e-10:
        mem = (mem - float(np.mean(mem))) / m_std  # unit Gaussian AR(1)

    # Step 2: quantile map ranks → skewnorm, then theoretical normalize
    z_skew       = _quantile_map(mem, skew_a)
    mu_sn, std_sn = _skewnorm_theoretical_moments(skew_a)
    z_sn         = (z_skew - mu_sn) / std_sn      # mean≈0, std≈1, skew preserved

    # Step 3: Student-t fat tails, blend 60/40 with skewnorm shape
    t_raw = stats.t.rvs(df=max(df_t, 2.01), loc=0, scale=1,
                        size=n_bars, random_state=rng)
    t_m, t_s = float(np.mean(t_raw)), float(np.std(t_raw))
    if t_s > 1e-10:
        t_std = (t_raw - t_m) / t_s
    else:
        t_std = t_raw

    z_blend = 0.6 * z_sn + 0.4 * t_std
    zb_m, zb_s = float(np.mean(z_blend)), float(np.std(z_blend))
    if zb_s > 1e-10:
        z_final = (z_blend - zb_m) / zb_s
    else:
        z_final = z_blend

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
