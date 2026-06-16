"""
stat_process.py  v18
====================
純統計過程模型，完全不使用 agent。

Fix-18 — 三項修改
-----------------
修改一：soft anchor（修復價格路徑漂移）
  real_anchor_weight 預設改為 0.3，對每個 window 的起始價格做幾何加權：
    start_px = exp((1-w)*log(sim_last) + w*log(real_close))
  另外在 generate() 內部加入 drift_correction：
    計算 sim 序列最後一個 close 與下一個真實 window 開始價格的 log 差距，
    分攤到 n_bars 根 bar 上作為額外的 mu_correction，避免誤差無限累積。
  注意：drift_correction 在 rolling_fit_generate 中由 next_real_open 傳入。

修改二：mixture model（修復 skew 翻轉 + 提升 kurtosis）
  拋棄單純 rank-remap，改用概率混合：
    p_t = clip(4.0 / df_t, 0.1, 0.6)   # df 越低，t 成分越重
    mask ~ Bernoulli(p_t)
    z_mix = where(mask, t_sample, skewnorm_sample)
  skew 方向由 skewnorm 成分（40-90%）決定，kurtosis 由 t 成分拉高。
  混合後統一 rescale 回 (ret_mu, ret_std)。
  AR(1) copula 邏輯保持不變（對 z_mix 做 rank-remap）。

修改三：trend bias（方向命中率微調）
  在 fit() 擷取的 ret_mu 基礎上，加入前一個 lookback 真實走勢的 sign bias：
    real_trend = sum(log_returns of lookback window)
    trend_bias = sign(real_trend) * abs(ret_mu) * 0.3
    params["ret_mu"] = ret_mu + trend_bias
  邏輯上延續 momentum，不影響分佈的 std/skew/kurtosis。

v1-v18 修正歷程
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
  Fix-16-debug: 定位到 amp 乘法在 PPF 階段就已翻轉 skew
  Fix-17  : 雙層 rank-remap（skewnorm rank onto t samples）+ AR(1) copula
             => 結構正確，但 kurtosis 仍偏低、price path 向上漂移
  Fix-18  : soft anchor + mixture model + trend bias
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


# ---------------------------------------------------------------------------
# 1. FIT
# ---------------------------------------------------------------------------

def fit(df_history: pd.DataFrame, apply_trend_bias: bool = True) -> StatParams:
    """
    Fit stat params from df_history.
    apply_trend_bias: if True, nudge ret_mu in the direction of the overall
    lookback trend (Fix-18 修改三).
    """
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

    # Fix-18 修改三：trend bias
    if apply_trend_bias:
        real_trend = float(np.sum(log_rets))
        trend_bias = float(np.sign(real_trend) * abs(ret_mu) * 0.3)
        ret_mu = ret_mu + trend_bias

    return StatParams(
        ret_mu       = ret_mu,
        ret_std      = ret_std,
        ret_skew_a   = skew_a,
        ret_df       = df_t,
        # hurst clip [0.3, 0.65]: h=0.65 => AR(1) rho=0.21
        hurst_target = float(np.clip(h, 0.3, 0.65)),
        wick_lambda  = wick_lam,
        atr_mean     = atr_mean,
    )


# ---------------------------------------------------------------------------
# 2. GENERATE  (Fix-18: mixture model + drift correction)
# ---------------------------------------------------------------------------

def _ar1_hurst_rho(h: float) -> float:
    return float(np.clip(2 ** (2 * h - 1) - 1, -0.95, 0.95))


def generate(
    params:           StatParams,
    n_bars:           int,
    start_price:      float = 100.0,
    seed:             int | None = None,
    drift_correction: float = 0.0,
) -> pd.DataFrame:
    """
    Fix-18: mixture model for clean skew + kurtosis, plus optional drift correction.

    Step 1 — mixture sampling
      p_t = clip(4 / df_t, 0.1, 0.6)   (df 越低 => 更多 t 成分)
      mask ~ Bernoulli(p_t)
      z_mix = where(mask, t_sample, skewnorm_sample)
      => skew 由 skewnorm 決定，kurtosis 由 t 拉高

    Step 2 — AR(1) copula
      AR(1) ranks imposed on z_mix (same as v17).

    Step 3 — rescale to (ret_mu + drift_correction, ret_std)

    drift_correction: 每 bar 的額外 mu，用來消化與真實下一個 window
    開始價格的 log 差距，避免路徑漂移（Fix-18 修改一的 generate 端）。
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
    # Step 1: mixture sampling
    # ------------------------------------------------------------------
    p_t   = float(np.clip(4.0 / max(df_t, 2.01), 0.1, 0.6))
    mask  = rng.uniform(size=n_bars) < p_t

    sn_samples = stats.skewnorm.rvs(a=skew_a, loc=0, scale=1,
                                    size=n_bars, random_state=rng)
    t_samples  = stats.t.rvs(df=max(df_t, 2.01), loc=0, scale=1,
                              size=n_bars, random_state=rng)
    z_mix = np.where(mask, t_samples, sn_samples)

    # ------------------------------------------------------------------
    # Step 2: AR(1) copula (rank-remap onto z_mix)
    # ------------------------------------------------------------------
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

    rank_ar1       = np.argsort(np.argsort(u_ar1))
    z_mix_sorted   = np.sort(z_mix)
    z_final        = z_mix_sorted[rank_ar1]

    # ------------------------------------------------------------------
    # Step 3: single linear rescale to (ret_mu, ret_std)
    # ------------------------------------------------------------------
    z_mean = float(np.mean(z_final))
    z_std  = float(np.std(z_final))
    if z_std > 1e-10:
        log_rets = (z_final - z_mean) / z_std * ret_std + ret_mu
    else:
        log_rets = np.full(n_bars, ret_mu)

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
    real_anchor_weight:  float = 0.3,   # Fix-18: 預設改為 0.3
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

        # Fix-18 修改一: soft anchor（幾何加權起始價格）
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

        # Fix-18 修改一: drift_correction
        # 計算「若模擬要在 actual_fwd 根 bar 後到達真實下一個 window 的開始價格」
        # 需要的每 bar 額外 mu（lookahead 一個 step 的真實 open）
        next_real_idx = min(fwd_end, n_total - 1)
        next_real_open = float(df_real["Open"].iloc[next_real_idx])
        if actual_fwd > 0 and start_px > 0 and next_real_open > 0:
            log_gap = np.log(next_real_open / start_px)
            # 分攤一半的差距（不要完全追，保留隨機性）
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

        # -------------------------------------------------------------------
        # Loss calculation
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # OHLC comparison log（Fix-18 新增）
        # 對照真實 forward window 的 OHLC 摘要 vs 模擬的 OHLC 摘要
        # -------------------------------------------------------------------
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
            # OHLC comparison
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
