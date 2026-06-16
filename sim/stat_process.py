"""
stat_process.py  v22
====================
純統計過程模型，完全不使用 agent。

架構：回退到 v18 穩定骨架，新增三項市場微觀特徵
------------------------------------------------------

v18 保留不動：
  - mixture model: p_t 混合 t(df) + skewnorm(a)
  - AR(1) copula rank-remap（hurst → rho）
  - single linear rescale to (ret_mu, ret_std)
  - soft anchor (real_anchor_weight=0.3)
  - drift_correction（消化 window 間的 log 差距）
  - trend_bias（延續前一 lookback 的動量方向）

v22 新增三項市場微觀特徵：
------------------------------------------------------

特徵 A：Jump Component（解決 kurtosis 不足）
  真實市場 kurtosis 10.6 主要來自跳空（earnings/macro），
  不是連續 t 分布能模擬的。
  fit()  新增：
    jump_freq = count(|r| > 3*ret_std) / n           # 跳空機率
    jump_std  = std(r where |r| > 3*ret_std) or 3*ret_std
  generate() 新增（在 rescale 之後）：
    jump_mask = Bernoulli(jump_freq)
    jump_sizes = Normal(0, jump_std)
    log_rets += where(jump_mask, jump_sizes, 0)
    最後做一次 std rescale 保持整體 volatility 不膨脹

特徵 B：GARCH-like Vol Clustering（解決序列相關性）
  真實市場大波動後往往跟著大波動（波動叢聚）。
  fit()  新增：
    vol_persistence = corr(resid_sq[:-1], resid_sq[1:])  # ARCH(1) alpha
    vol_persistence = clip(val, 0.0, 0.85)
  generate() 新增（取代固定 ret_std）：
    h_t[0] = ret_std^2
    h_t[i] = (1 - alpha) * ret_std^2 + alpha * (log_ret[i-1])^2
    innov[i] = z_final[i] * sqrt(h_t[i]) / ret_std   # 保持均值 std 不變

特徵 C：ACF Lag-1 短期均值回歸 vs Momentum 分離
  hurst 只捕捉長程依賴，1-bar ACF 反映短期 mean-reversion/momentum。
  fit()  新增：
    acf_lag1 = corr(log_rets[:-1], log_rets[1:])   # 1-bar autocorrelation
  generate() 新增：
    在 AR(1) copula 的 rho 之外，額外對 z_final 加入
    z_final[i] += acf_lag1 * z_final[i-1] * 0.3    # dampen 0.3 避免爆炸
    （只在 |acf_lag1| > 0.05 時啟用）

v1-v22 修正歷程
--------------
  Fix-1~3 : df 掃描、skewnorm、rolling ATR wick
  Fix-4~8 : AR(1) 正規化、mean offset、rolling anchor
  Fix-9~11: 失敗—線性 t-blend 消除 skew
  Fix-12  : Gaussian Copula + skewnorm 邊際 => skew 修復但 kurtosis≈2.5
  Fix-13  : quantile-blend => kurtosis 4.78 但 skew 翻轉 -0.82
  Fix-14  : Tail Amplifier => kurtosis↑ 但 skew 仍翻轉
  Fix-15  : center-masked amp + variance-mixture => kurtosis 9.74 但 std 偏高
  Fix-16  : symmetric clip + std rescale => std✅ hurst✅ skew仍翻轉
  Fix-17  : 雙層 rank-remap => 結構正確但 kurtosis 偏低
  Fix-18  : soft anchor + mixture model + trend bias => hurst✅ 方向命中率✅
  Fix-19  : GARCH + directed t-mixture => skew 接近 0 但 kurtosis 1.84
  Fix-20  : chi2 tail amp + AR(1) direct => kurtosis 1.92（chi2 被 rescale 抵消）
  Fix-21  : chi2 tail amp 移到 rescale 後 + skew_mean_contribution
             => kurtosis 8.98✅ hurst 0.705✅ 但 std 膨脹 0.025、skew -0.83
  Fix-22  : 回退 v18 骨架 + jump component + GARCH vol clustering + ACF lag-1
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
    ret_mu:           float
    ret_std:          float
    ret_skew_a:       float
    ret_df:           float
    hurst_target:     float
    wick_lambda:      float
    atr_mean:         float
    # v22 新增
    jump_freq:        float
    jump_std:         float
    vol_persistence:  float
    acf_lag1:         float


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


def _fit_jump_params(log_rets: np.ndarray, ret_std: float) -> tuple[float, float]:
    """
    特徵 A：Jump Component fitting
    跳空定義：|r| > 3 * ret_std
    """
    threshold  = 3.0 * ret_std
    jump_mask  = np.abs(log_rets) > threshold
    jump_count = int(np.sum(jump_mask))
    jump_freq  = float(jump_count) / max(len(log_rets), 1)
    if jump_count >= 2:
        jump_std = float(np.std(log_rets[jump_mask]))
    else:
        jump_std = float(ret_std * 3.0)
    jump_std = max(jump_std, ret_std * 2.0)  # 下限：至少 2x 日常波動
    return jump_freq, jump_std


def _fit_vol_persistence(log_rets: np.ndarray, ret_mu: float) -> float:
    """
    特徵 B：GARCH-like vol clustering
    用 ARCH(1) correlation 代理 vol_persistence
    """
    resid    = log_rets - ret_mu
    resid_sq = resid ** 2
    if len(resid_sq) < 4:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = float(np.corrcoef(resid_sq[:-1], resid_sq[1:])[0, 1])
    return float(np.clip(corr if np.isfinite(corr) else 0.0, 0.0, 0.85))


def _fit_acf_lag1(log_rets: np.ndarray) -> float:
    """
    特徵 C：1-bar autocorrelation
    正值 = momentum，負值 = mean-reversion
    """
    if len(log_rets) < 4:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = float(np.corrcoef(log_rets[:-1], log_rets[1:])[0, 1])
    return float(np.clip(corr if np.isfinite(corr) else 0.0, -0.5, 0.5))


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

    # v22 新增特徵
    jump_freq, jump_std  = _fit_jump_params(log_rets, ret_std)
    vol_persistence      = _fit_vol_persistence(log_rets, ret_mu)
    acf_lag1             = _fit_acf_lag1(log_rets)

    # trend bias（Fix-18 保留）
    if apply_trend_bias:
        real_trend = float(np.sum(log_rets))
        trend_bias = float(np.sign(real_trend) * abs(ret_mu) * 0.3)
        ret_mu = ret_mu + trend_bias

    return StatParams(
        ret_mu          = ret_mu,
        ret_std         = ret_std,
        ret_skew_a      = skew_a,
        ret_df          = df_t,
        hurst_target    = float(np.clip(h, 0.3, 0.65)),
        wick_lambda     = wick_lam,
        atr_mean        = atr_mean,
        jump_freq       = jump_freq,
        jump_std        = jump_std,
        vol_persistence = vol_persistence,
        acf_lag1        = acf_lag1,
    )


# ---------------------------------------------------------------------------
# 2. GENERATE  (v22: v18 base + jump + GARCH vol + ACF lag-1)
# ---------------------------------------------------------------------------

def generate(
    params:           StatParams,
    n_bars:           int,
    start_price:      float = 100.0,
    seed:             int | None = None,
    drift_correction: float = 0.0,
) -> pd.DataFrame:
    """
    v22 generate pipeline:

    Step 1 — mixture sampling (v18 不變)
      p_t = clip(4 / df_t, 0.1, 0.6)
      z_mix = where(Bernoulli(p_t), t_sample, skewnorm_sample)

    Step 2 — AR(1) copula rank-remap (v18 不變)
      rho = 2^(2h-1) - 1
      u_ar1 = AR(1)(rho, eps)
      z_final = z_mix_sorted[rank(u_ar1)]

    Step 3 — ACF lag-1 微調 (v22 新增特徵 C)
      若 |acf_lag1| > 0.05：
        z_adj[i] = z_final[i] + acf_lag1 * z_final[i-1] * 0.3

    Step 4 — GARCH-like vol clustering (v22 新增特徵 B)
      h_t[0] = ret_std^2
      h_t[i] = (1-alpha)*ret_std^2 + alpha*z_adj[i-1]^2
      z_garch[i] = z_adj[i] * sqrt(h_t[i]) / ret_std

    Step 5 — rescale to (ret_mu + drift_correction, ret_std) (v18 不變)

    Step 6 — Jump injection (v22 新增特徵 A)
      jump_mask = Bernoulli(jump_freq)
      log_rets += where(jump_mask, Normal(0, jump_std), 0)
      最後 rescale 保持 std 不膨脹
    """
    rng = np.random.default_rng(seed)

    ret_mu          = params["ret_mu"] + drift_correction
    ret_std         = params["ret_std"]
    skew_a          = params["ret_skew_a"]
    df_t            = params["ret_df"]
    hurst           = params["hurst_target"]
    wick_lam        = params["wick_lambda"]
    atr_mean        = params["atr_mean"]
    jump_freq       = params["jump_freq"]
    jump_std        = params["jump_std"]
    vol_persistence = params["vol_persistence"]
    acf_lag1        = params["acf_lag1"]

    # ------------------------------------------------------------------
    # Step 1: mixture sampling (v18)
    # ------------------------------------------------------------------
    p_t  = float(np.clip(4.0 / max(df_t, 2.01), 0.1, 0.6))
    mask = rng.uniform(size=n_bars) < p_t

    sn_samples = stats.skewnorm.rvs(a=skew_a, loc=0, scale=1,
                                    size=n_bars, random_state=rng)
    t_samples  = stats.t.rvs(df=max(df_t, 2.01), loc=0, scale=1,
                              size=n_bars, random_state=rng)
    z_mix = np.where(mask, t_samples, sn_samples)

    # ------------------------------------------------------------------
    # Step 2: AR(1) copula rank-remap (v18)
    # ------------------------------------------------------------------
    rho = _ar1_hurst_rho(hurst)
    eps = rng.standard_normal(n_bars)
    if abs(rho) > 1e-6:
        u_ar1    = np.empty(n_bars)
        u_ar1[0] = eps[0]
        innov_sc = float(np.sqrt(max(1.0 - rho ** 2, 0.0)))
        for i in range(1, n_bars):
            u_ar1[i] = rho * u_ar1[i-1] + innov_sc * eps[i]
    else:
        u_ar1 = eps.copy()

    rank_ar1      = np.argsort(np.argsort(u_ar1))
    z_mix_sorted  = np.sort(z_mix)
    z_final       = z_mix_sorted[rank_ar1]

    # ------------------------------------------------------------------
    # Step 3: ACF lag-1 微調 (v22 特徵 C)
    # ------------------------------------------------------------------
    if abs(acf_lag1) > 0.05 and n_bars > 1:
        _LAG1_DAMP = 0.3
        z_adj    = z_final.copy()
        for i in range(1, n_bars):
            z_adj[i] = z_final[i] + acf_lag1 * z_final[i-1] * _LAG1_DAMP
        # 重新 normalize 保持 z 的 std=1
        z_std = float(np.std(z_adj))
        if z_std > 1e-10:
            z_adj = z_adj / z_std
    else:
        z_adj = z_final

    # ------------------------------------------------------------------
    # Step 4: GARCH-like vol clustering (v22 特徵 B)
    # ------------------------------------------------------------------
    alpha = vol_persistence  # ARCH(1) coefficient
    if alpha > 0.01 and n_bars > 1:
        base_var  = ret_std ** 2
        h_t       = np.empty(n_bars)
        h_t[0]    = base_var
        z_garch   = np.empty(n_bars)
        z_garch[0] = z_adj[0]
        for i in range(1, n_bars):
            prev_ret  = z_garch[i-1] * ret_std  # 近似上一根實際 return
            h_t[i]    = (1.0 - alpha) * base_var + alpha * prev_ret ** 2
            vol_scale = float(np.sqrt(max(h_t[i], 1e-12)) / (ret_std + 1e-10))
            vol_scale = float(np.clip(vol_scale, 0.3, 3.0))
            z_garch[i] = z_adj[i] * vol_scale
    else:
        z_garch = z_adj

    # ------------------------------------------------------------------
    # Step 5: single linear rescale to (ret_mu, ret_std) (v18)
    # ------------------------------------------------------------------
    z_mean = float(np.mean(z_garch))
    z_std  = float(np.std(z_garch))
    if z_std > 1e-10:
        z_scaled = (z_garch - z_mean) / z_std * ret_std + ret_mu
    else:
        z_scaled = np.full(n_bars, ret_mu)

    log_rets = z_scaled.copy()

    # ------------------------------------------------------------------
    # Step 6: Jump injection (v22 特徵 A)
    # 在 rescale 之後注入，不被標準化消除
    # ------------------------------------------------------------------
    if jump_freq > 0 and n_bars > 0:
        jump_mask  = rng.uniform(size=n_bars) < jump_freq
        jump_sizes = rng.normal(0.0, jump_std, size=n_bars)
        log_rets   = log_rets + np.where(jump_mask, jump_sizes, 0.0)

        # 保持整體 std 不因 jump 膨脹：把 jump 貢獻分離出來再 rescale
        body_std = float(np.std(log_rets - ret_mu))
        if body_std > 1e-10 and body_std > ret_std * 1.05:
            log_rets = (log_rets - ret_mu) * (ret_std / body_std) + ret_mu

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
            "drift_corr":      round(drift_corr, 6),
            "loss":            round(loss, 4),
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
                f"jfreq={params['jump_freq']:.3f}  "
                f"vp={params['vol_persistence']:.3f}  "
                f"acf1={params['acf_lag1']:+.3f}  "
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
