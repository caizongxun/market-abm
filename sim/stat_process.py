"""
stat_process.py  v21
====================
純統計過程模型，完全不使用 agent。

Fix-21 — 修正 v20 三個根因
--------------------------------------
根因一：chi2 tail amplifier 被 rescale 抵消（kurtosis 無效）
  v20 在 rescale 之前做 chi2 放大，最後的 z-score rescale 把尾部壓回去了。
  v21 把 chi2 tail amplifier 移到 rescale 之後（Step 5），
  放大後不再被標準化消除，kurtosis 自然保留。

根因二：全 z-score 標準化消除了 skew 的 mean offset（skew 符號丟失）
  skewnorm 的 mean offset 是 skew 方向的訊號，
  全 z-score (z - mean) / std 把這個訊號清零。
  v21 標準化後把 skew_mean_contribution 還原：
    skew_offset = z_mean_mix / (z_std_mix + 1e-10)
    skew_mean_contribution = skew_offset * ret_std * 0.5  (dampen 0.5 防漂移)
  最後 log_rets = z_scaled + ret_mu + skew_mean_contribution

根因三：skew_a 放大倍數不足（skew 仍偏負）
  保留 v20 的 skew_a * 1.5 amplifier。

Fix-18 保留：soft anchor, drift_correction, trend_bias
Fix-19 保留：directed t-mixture (|t| * sign(skew_a))
Fix-20 保留：AR(1) direct recursion

v1-v21 修正歷程
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
             => std ✅  hurst ✅  skew 仍 -0.307 ❌  kurtosis 4.54 ❌
  Fix-17  : 雙層 rank-remap + AR(1) copula => kurtosis 偏低、price 漂移
  Fix-18  : soft anchor + mixture model + trend bias
             => hurst ✅  方向命中率 ✅  kurtosis 倒退至 1.94 ❌
  Fix-19  : GARCH(1,1) + directed t-mixture
             => skew -0.104 (接近 0) ✅  kurtosis 1.84 ❌  hurst 0.638 ❌
  Fix-20  : chi2 tail amp + skew_a*1.5 + AR(1) direct recursion
             => kurtosis 1.92 ❌（chi2 被 rescale 抵消）skew -0.167 ❌
  Fix-21  : chi2 tail amp 移到 rescale 後 + skew_mean_contribution
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


def _ar1_hurst_rho(h: float) -> float:
    return float(np.clip(2 ** (2 * h - 1) - 1, -0.95, 0.95))


# ---------------------------------------------------------------------------
# 1. FIT
# ---------------------------------------------------------------------------

def fit(df_history: pd.DataFrame, apply_trend_bias: bool = True) -> StatParams:
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

    # trend bias（Fix-18 保留）
    if apply_trend_bias:
        real_trend = float(np.sum(log_rets))
        trend_bias = float(np.sign(real_trend) * abs(ret_mu) * 0.3)
        ret_mu = ret_mu + trend_bias

    return StatParams(
        ret_mu       = ret_mu,
        ret_std      = ret_std,
        ret_skew_a   = skew_a,
        ret_df       = df_t,
        hurst_target = float(np.clip(h, 0.3, 0.65)),
        wick_lambda  = wick_lam,
        atr_mean     = atr_mean,
    )


# ---------------------------------------------------------------------------
# 2. GENERATE  (Fix-21)
# ---------------------------------------------------------------------------

# skew amplifier：補償 t 成分對 skewnorm 偏態的稀釋
_SKEW_AMP = 1.5

# chi2 tail amplifier threshold（以 ret_std 為單位）
_TAIL_THRESHOLD_SIGMA = 1.5

# skew mean contribution dampen factor（防止 price 漂移）
_SKEW_DAMPEN = 0.5


def generate(
    params:           StatParams,
    n_bars:           int,
    start_price:      float = 100.0,
    seed:             int | None = None,
    drift_correction: float = 0.0,
) -> pd.DataFrame:
    """
    Fix-21 generate pipeline:

    Step 1 — directed t-mixture with amplified skew_a
      skew_a_amp = skew_a * 1.5
      z_mix = where(mask, |t|*sign(skew_a), skewnorm(skew_a_amp))

    Step 2 — z-score normalize，保留 skew_mean_contribution
      z_mean_mix = mean(z_mix)
      z_norm = (z_mix - z_mean_mix) / std(z_mix)
      skew_offset = z_mean_mix / std(z_mix)
      skew_mean_contribution = skew_offset * ret_std * 0.5

    Step 3 — AR(1) direct recursion
      z_ar[i] = rho * z_ar[i-1] + innov_scale * z_norm[i]

    Step 4 — rescale to ret_std（只 scale，不 z-score）
      z_scaled = z_ar / std(z_ar) * ret_std

    Step 5 — chi2 tail amplifier（在 rescale 之後，kurtosis 不被消除）
      chi2_scale = clip(sqrt(nu / chi2(nu)), 0.5, 4.0)
      tail_mask = |z_scaled| > 1.5 * ret_std
      log_rets = where(tail_mask, z_scaled * chi2_scale, z_scaled)
                 + ret_mu + skew_mean_contribution
    """
    rng = np.random.default_rng(seed)

    ret_mu   = params["ret_mu"] + drift_correction
    ret_std  = params["ret_std"]
    skew_a   = params["ret_skew_a"]
    df_t     = params["ret_df"]
    hurst    = params["hurst_target"]
    wick_lam = params["wick_lambda"]
    atr_mean = params["atr_mean"]

    # ------------------------------------------------------------------
    # Step 1: directed t-mixture with amplified skew_a
    # ------------------------------------------------------------------
    skew_a_amp = float(np.clip(skew_a * _SKEW_AMP, -10.0, 10.0))
    p_t  = float(np.clip(4.0 / max(df_t, 2.01), 0.1, 0.6))
    mask = rng.uniform(size=n_bars) < p_t

    sn = stats.skewnorm.rvs(a=skew_a_amp, loc=0, scale=1,
                             size=n_bars, random_state=rng)
    t_raw = stats.t.rvs(df=max(df_t, 2.01), loc=0, scale=1,
                         size=n_bars, random_state=rng)

    if abs(skew_a) > 0.3:
        t_directed = np.abs(t_raw) * float(np.sign(skew_a))
    else:
        t_directed = t_raw

    z_mix = np.where(mask, t_directed, sn)

    # ------------------------------------------------------------------
    # Step 2: z-score normalize，但保留 skew mean offset
    # ------------------------------------------------------------------
    z_mean_mix = float(np.mean(z_mix))
    z_std_mix  = float(np.std(z_mix))
    if z_std_mix > 1e-10:
        z_norm      = (z_mix - z_mean_mix) / z_std_mix
        skew_offset = z_mean_mix / z_std_mix
    else:
        z_norm      = z_mix.copy()
        skew_offset = 0.0

    # skew 方向的 mean 貢獻（dampen 防止 price 漂移）
    skew_mean_contribution = skew_offset * ret_std * _SKEW_DAMPEN

    # ------------------------------------------------------------------
    # Step 3: AR(1) direct recursion
    # ------------------------------------------------------------------
    rho         = _ar1_hurst_rho(hurst)
    innov_scale = float(np.sqrt(max(1.0 - rho ** 2, 0.0)))

    z_ar    = np.empty(n_bars)
    z_ar[0] = z_norm[0]
    for i in range(1, n_bars):
        z_ar[i] = rho * z_ar[i - 1] + innov_scale * z_norm[i]

    # ------------------------------------------------------------------
    # Step 4: rescale to ret_std（只 scale，不做完整 z-score）
    # ------------------------------------------------------------------
    z_ar_std = float(np.std(z_ar))
    if z_ar_std > 1e-10:
        z_scaled = z_ar / z_ar_std * ret_std
    else:
        z_scaled = np.full(n_bars, 0.0)

    # ------------------------------------------------------------------
    # Step 5: chi2 tail amplifier（在 rescale 之後）
    # ------------------------------------------------------------------
    nu         = max(float(df_t), 3.0)
    chi2_draw  = rng.chisquare(nu, size=n_bars)
    chi2_scale = np.clip(np.sqrt(nu / np.maximum(chi2_draw, 1e-8)), 0.5, 4.0)

    tail_mask = np.abs(z_scaled) > _TAIL_THRESHOLD_SIGMA * ret_std
    z_tailed  = np.where(tail_mask, z_scaled * chi2_scale, z_scaled)

    log_rets = z_tailed + ret_mu + skew_mean_contribution

    # ------------------------------------------------------------------
    # Rebuild OHLC
    # ------------------------------------------------------------------
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
    real_anchor_weight:  float = 0.3,
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
        params    = fit(df_window, apply_trend_bias=True)

        # soft anchor（Fix-18 保留）
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

        # drift_correction（Fix-18 保留）
        next_real_idx  = min(fwd_end, n_total - 1)
        next_real_open = float(df_real["Open"].iloc[next_real_idx])
        if actual_fwd > 0 and start_px > 0 and next_real_open > 0:
            log_gap    = np.log(next_real_open / start_px)
            drift_corr = float(log_gap * 0.5 / actual_fwd)
        else:
            drift_corr = 0.0

        df_chunk = generate(
            params=params, n_bars=actual_fwd,
            start_price=start_px, seed=seed + window_idx,
            drift_correction=drift_corr,
        )
        sim_chunks.append(df_chunk)
        sim_last_close = float(df_chunk["Close"].iloc[-1])

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

        # OHLC comparison log
        real_o = float(real_fwd["Open"].iloc[0])   if len(real_fwd) > 0 else float("nan")
        real_h = float(real_fwd["High"].max())      if len(real_fwd) > 0 else float("nan")
        real_l = float(real_fwd["Low"].min())       if len(real_fwd) > 0 else float("nan")
        real_c = float(real_fwd["Close"].iloc[-1])  if len(real_fwd) > 0 else float("nan")
        sim_o  = float(df_chunk["Open"].iloc[0])
        sim_h  = float(df_chunk["High"].max())
        sim_l  = float(df_chunk["Low"].min())
        sim_c  = float(df_chunk["Close"].iloc[-1])

        def _pct(a, b):
            return round((a - b) / max(abs(b), 1e-8) * 100, 2) if b == b else float("nan")

        param_log.append({
            "window":   window_idx + 1,
            "fit_bars": [fit_start, fit_end],
            "fwd_bars": [pos, fwd_end],
            **{k: params[k] for k in params},
            "drift_corr": round(drift_corr, 6),
            "loss":     round(loss, 4),
            "ohlc_real": {"O": round(real_o,2), "H": round(real_h,2),
                          "L": round(real_l,2), "C": round(real_c,2)},
            "ohlc_sim":  {"O": round(sim_o,2),  "H": round(sim_h,2),
                          "L": round(sim_l,2),  "C": round(sim_c,2)},
            "ohlc_err_pct": {
                "O": _pct(sim_o, real_o), "H": _pct(sim_h, real_h),
                "L": _pct(sim_l, real_l), "C": _pct(sim_c, real_c),
            },
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
            print(
                f"         real OHLC  O={real_o:>8.2f}  H={real_h:>8.2f}  "
                f"L={real_l:>8.2f}  C={real_c:>8.2f}"
            )
            print(
                f"         sim  OHLC  O={sim_o:>8.2f}  H={sim_h:>8.2f}  "
                f"L={sim_l:>8.2f}  C={sim_c:>8.2f}  "
                f"(C err={_pct(sim_c, real_c):+.1f}%)"
            )

        pos += step
        window_idx += 1

    if not sim_chunks:
        raise RuntimeError("No chunks generated -- check lookback/step settings.")

    return pd.concat(sim_chunks, ignore_index=True), param_log
