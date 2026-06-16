"""
stat_process.py  v36
====================
純統計過程模型，完全不使用 agent。

v36 fix（v35 骨架不動）：
------------------------------------------------------
Fix-O v36：final_scale scale_max 改為與 target_ek 掛鉤，不再依賴 skew 符號
  原邏輯：scale_max = 1.15 if skew_a > 0.3 else 1.0
    → skew 為負時 chi2 尾部空間被鎖死在 1.0，target_ek 高也無效。
  新邏輯：scale_max = clip(1.0 + (target_ek - 3.0) * 0.02, 1.0, 1.4)
    target_ek=3  → 1.00（不膨脹）
    target_ek=10 → 1.14（任何 skew 方向都能讓尾部呼吸）
    target_ek=15 → 1.24
    target_ek=28 → 1.40（上限）

v1-v36 修正歷程
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
  Fix-20  : chi2 tail amp + AR(1) direct => kurtosis 1.92
  Fix-21  : chi2 tail amp 移到 rescale 後 => kurtosis 8.98 但 std 膨脹、skew -0.83
  Fix-22  : 回退 v18 + jump + GARCH + ACF => hurst✅ 但 kurtosis 2.02、skew -0.30
  Fix-23  : additive AR(1) + body-only rescale + kurtosis-driven p_t
             => kurtosis 3.97✅↑ hurst 0.74 偏高 skew -0.24
  Fix-24  : AR(1) warmup + global ek floor + hurst clip 0.69
             => skew -0.068✅ hurst 0.720✅ kurtosis 2.62 偏低
  Fix-25  : OnlineRidgePredictor => skew +0.023✅ hurst 0.717✅ 方向命中率 0.516✅
             kurtosis 2.90 退步
  Fix-26  : variance-mixture tail booster + target_ek in predictor
             => kurtosis 5.43✅↑ hurst 0.719✅ std 膨脹 1.858 skew -0.203 仍偏負
  Fix-27  : Fix-A global std rescale + Fix-B aggressive nu_boost +
             Fix-C skew sign protection
             => kurtosis 4.85 退步（Fix-A 把尾部壓回去了）
             skew +0.741✅ hurst 0.690✅ std 1.763
  Fix-28  : Fix-A 改為 body-only rescale（scale_a=0.85），不碰尾部
             => kurtosis 4.07 退步（兩次 body rescale 疊加）
  Fix-29  : 移除 chi2 boost 內的舊 body rescale，只留 Fix-A 的單次縮放
             => kurtosis 4.32↑ hurst 0.690✅ skew +0.712✅ std 1.812 偏高
             方向命中率 0.519✅
  Fix-30  : chi2 更激進 + 全局 std 收斂(z_scaled) + skew 幅度限制
             => kurtosis 8.82✅↑ skew +0.466✅ hurst 0.678✅ std 1.810 仍偏高
             方向命中率 0.539✅
  Fix-31  : 移除 jump 後 body rescale，改為 jump 後 final global std 收斂
             => kurtosis 13.03 過衝 skew +0.188✅ hurst 0.681✅ std 1.785 仍偏高
             方向命中率 0.539✅
  Fix-32  : chi2 clip 5.0→3.0 抑制 kurtosis 過衝
             final_scale clip (0.6,1.4)→(0.5,1.0) 只允許縮
             => kurtosis 5.64 skew +0.152 hurst 0.685✅ std 1.662✅
             方向命中率 0.529✅
  Fix-33  : chi2 clip 3.0→4.0 繼續推 kurtosis
             final_scale skew-adaptive：skew_a>0.3 → max 1.15，else max 1.0
             新增 DTW distance + path_corr 走勢相似性指標（不進 loss，只記錄）
             => kurtosis 7.62 skew +0.172 hurst 0.682✅ std 1.672✅
             方向命中率 0.535✅
  Fix-34  : Fix-I tail_threshold 1.5σ→1.2σ + chi2 clip 4.0→3.5
             Fix-J global_scale skew-aware（正尾允許微幅放大至 1.05）
             Fix-K nu_boost 改為與 df_t 反比：6/(df_t-2) clip(1.5,5.0)
             => kurtosis 4.08 skew +0.607✅ hurst 0.694✅ std 1.650✅
             方向命中率 0.519✅
             (Fix-K 實際上 clip 全在 1.5 下界，差異化失效)
  Fix-35  : Fix-L nu_boost = 3/(df_t-4) clip(0.5,3.0) 真正差異化
             Fix-M chi2 極端尾部（>2.5σ）允許 clip(0.5,5.0)，普通尾部仍 3.5
             Fix-N p_t clip 上限 0.85 → 0.92
  Fix-36  : Fix-O scale_max = clip(1.0+(target_ek-3)*0.02, 1.0, 1.4)
             不再因 skew 為負就鎖死 final_scale 上限在 1.0
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
    jump_freq:        float
    jump_std:         float
    vol_persistence:  float
    acf_lag1:         float
    target_ek:        float


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
    threshold  = 3.0 * ret_std
    jump_mask  = np.abs(log_rets) > threshold
    jump_count = int(np.sum(jump_mask))
    jump_freq  = float(jump_count) / max(len(log_rets), 1)
    if jump_count >= 2:
        jump_std = float(np.std(log_rets[jump_mask]))
    else:
        jump_std = float(ret_std * 3.0)
    jump_std = max(jump_std, ret_std * 2.0)
    return jump_freq, jump_std


def _fit_vol_persistence(log_rets: np.ndarray, ret_mu: float) -> float:
    resid    = log_rets - ret_mu
    resid_sq = resid ** 2
    if len(resid_sq) < 4:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = float(np.corrcoef(resid_sq[:-1], resid_sq[1:])[0, 1])
    return float(np.clip(corr if np.isfinite(corr) else 0.0, 0.0, 0.85))


def _fit_acf_lag1(log_rets: np.ndarray) -> float:
    if len(log_rets) < 4:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = float(np.corrcoef(log_rets[:-1], log_rets[1:])[0, 1])
    return float(np.clip(corr if np.isfinite(corr) else 0.0, -0.5, 0.5))


def _ar1_hurst_rho(h: float) -> float:
    return float(np.clip(2 ** (2 * h - 1) - 1, -0.95, 0.95))


# ---------------------------------------------------------------------------
# Fix-H v33: 走勢相似性輔助函數
# ---------------------------------------------------------------------------

def _dtw_distance(s: np.ndarray, t: np.ndarray) -> float:
    n, m = len(s), len(t)
    if n == 0 or m == 0:
        return float("nan")
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s[i - 1] - t[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    raw = dtw[n, m]
    return float(raw / max(n, m))


def _path_corr(real_closes: np.ndarray, sim_closes: np.ndarray) -> float:
    n = min(len(real_closes), len(sim_closes))
    if n < 3:
        return float("nan")
    r_path = np.log(np.maximum(real_closes[:n], 1e-10) / max(real_closes[0], 1e-10))
    s_path = np.log(np.maximum(sim_closes[:n], 1e-10) / max(sim_closes[0], 1e-10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = float(np.corrcoef(r_path, s_path)[0, 1])
    return corr if np.isfinite(corr) else float("nan")


# ---------------------------------------------------------------------------
# 1. FIT
# ---------------------------------------------------------------------------

def fit(
    df_history:      pd.DataFrame,
    apply_trend_bias: bool = True,
    ek_global_floor:  float = 3.0,
) -> StatParams:
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

    jump_freq, jump_std = _fit_jump_params(log_rets, ret_std)
    vol_persistence     = _fit_vol_persistence(log_rets, ret_mu)
    acf_lag1            = _fit_acf_lag1(log_rets)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ek = float(stats.kurtosis(log_rets))
    target_ek = float(max(np.clip(ek, 0.5, 30.0), ek_global_floor))

    if apply_trend_bias:
        real_trend = float(np.sum(log_rets))
        trend_bias = float(np.sign(real_trend) * abs(ret_mu) * 0.3)
        ret_mu = ret_mu + trend_bias

    return StatParams(
        ret_mu          = ret_mu,
        ret_std         = ret_std,
        ret_skew_a      = skew_a,
        ret_df          = df_t,
        hurst_target    = float(np.clip(h, 0.3, 0.69)),
        wick_lambda     = wick_lam,
        atr_mean        = atr_mean,
        jump_freq       = jump_freq,
        jump_std        = jump_std,
        vol_persistence = vol_persistence,
        acf_lag1        = acf_lag1,
        target_ek       = target_ek,
    )


# ---------------------------------------------------------------------------
# 2. GENERATE  (v36: Fix-O)
# ---------------------------------------------------------------------------

_AR1_WARMUP = 50


def generate(
    params:           StatParams,
    n_bars:           int,
    start_price:      float = 100.0,
    seed:             int | None = None,
    drift_correction: float = 0.0,
) -> pd.DataFrame:
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
    target_ek       = params["target_ek"]

    t_ek = 6.0 / max(df_t - 4.0, 0.1)
    # Fix-N v35: p_t clip 上限 0.85 → 0.92
    p_t  = float(np.clip(target_ek / (t_ek + 1e-8), 0.10, 0.92))

    total = n_bars + _AR1_WARMUP
    mask       = rng.uniform(size=total) < p_t
    sn_samples = stats.skewnorm.rvs(a=skew_a, loc=0, scale=1,
                                    size=total, random_state=rng)
    t_samples  = stats.t.rvs(df=max(df_t, 2.01), loc=0, scale=1,
                              size=total, random_state=rng)
    z_mix_ext = np.where(mask, t_samples, sn_samples)

    rho = _ar1_hurst_rho(hurst)
    if abs(rho) > 1e-6:
        innov_sc    = float(np.sqrt(max(1.0 - rho ** 2, 0.0)))
        z_ar_ext    = np.empty(total)
        z_ar_ext[0] = z_mix_ext[0]
        for i in range(1, total):
            z_ar_ext[i] = rho * z_ar_ext[i - 1] + innov_sc * z_mix_ext[i]
        z_ar = z_ar_ext[_AR1_WARMUP:]
    else:
        z_ar = z_mix_ext[_AR1_WARMUP:].copy()

    z_final = z_ar

    if abs(acf_lag1) > 0.05 and n_bars > 1:
        z_adj = z_final.copy()
        for i in range(1, n_bars):
            z_adj[i] = z_final[i] + acf_lag1 * z_final[i - 1] * 0.3
        z_std = float(np.std(z_adj))
        if z_std > 1e-10:
            z_adj = z_adj / z_std
    else:
        z_adj = z_final

    alpha = vol_persistence
    if alpha > 0.01 and n_bars > 1:
        base_var   = ret_std ** 2
        h_t        = np.empty(n_bars)
        h_t[0]     = base_var
        z_garch    = np.empty(n_bars)
        z_garch[0] = z_adj[0]
        for i in range(1, n_bars):
            prev_ret   = z_garch[i - 1] * ret_std
            h_t[i]     = (1.0 - alpha) * base_var + alpha * prev_ret ** 2
            vol_scale  = float(np.clip(
                np.sqrt(max(h_t[i], 1e-12)) / (ret_std + 1e-10), 0.3, 3.0
            ))
            z_garch[i] = z_adj[i] * vol_scale
    else:
        z_garch = z_adj

    z_mean = float(np.mean(z_garch))
    z_std  = float(np.std(z_garch))
    if z_std > 1e-10:
        z_scaled = (z_garch - z_mean) / z_std * ret_std + ret_mu
    else:
        z_scaled = np.full(n_bars, ret_mu)

    # ------------------------------------------------------------------
    # Fix-I v34: tail_threshold 1.2σ
    # Fix-L v35: nu_boost = clip(3.0/max(df_t-4.0,0.1), 0.5, 3.0) — 真正差異化
    #   df_t=7  → 3/3=1.0  （本身已肥，輕推）
    #   df_t=10 → 3/6=0.5  → clip → 0.5（更肥的 chi2）
    #   df_t=20 → 0.19     → clip → 0.5
    #   df_t=30 → 0.12     → clip → 0.5（接近 Normal，最需要強尾）
    # Fix-M v35: 普通 tail bar clip(0.5,3.5)，極端尾部(>2.5σ)允許 clip(0.5,5.0)
    # ------------------------------------------------------------------
    tail_threshold  = 1.2 * ret_std
    tail_mask_boost = np.abs(z_scaled - ret_mu) > tail_threshold

    if tail_mask_boost.sum() >= 2:
        # Fix-L: 反比公式，df_t 越大 nu 越小 → chi2 越肥
        nu_boost   = float(np.clip(3.0 / max(df_t - 4.0, 0.1), 0.5, 3.0))
        chi2_draws = rng.chisquare(nu_boost, size=int(tail_mask_boost.sum()))
        chi2_scale = np.sqrt(nu_boost / np.maximum(chi2_draws, 1e-8))

        # Fix-M: 分兩段 clip
        # 普通尾部 clip(0.5, 3.5)
        chi2_scale = np.clip(chi2_scale, 0.5, 3.5)
        # 極端尾部 (>2.5σ) 允許到 5.0
        tail_vals_for_mask = z_scaled[tail_mask_boost]
        extreme_mask = np.abs(tail_vals_for_mask - ret_mu) > 2.5 * ret_std
        chi2_scale[extreme_mask] = np.clip(chi2_scale[extreme_mask], 0.5, 5.0)

        z_boosted = z_scaled.copy()
        z_boosted[tail_mask_boost] = ret_mu + (tail_vals_for_mask - ret_mu) * chi2_scale

        z_scaled = z_boosted

    # ------------------------------------------------------------------
    # Fix-A v29: body-only rescale（單次），不碰尾部
    # ------------------------------------------------------------------
    body_mask_a = np.abs(z_scaled - ret_mu) <= tail_threshold
    if body_mask_a.sum() > 5:
        body_std_a = float(np.std(z_scaled[body_mask_a] - ret_mu))
        if body_std_a > 1e-8:
            scale_a = ret_std * 0.85 / body_std_a
            scale_a = float(np.clip(scale_a, 0.5, 2.0))
            z_scaled[body_mask_a] = ret_mu + (z_scaled[body_mask_a] - ret_mu) * scale_a

    # ------------------------------------------------------------------
    # Fix-J v34: global_scale skew-aware 等比縮放
    # ------------------------------------------------------------------
    cur_std = float(np.std(z_scaled - ret_mu))
    if cur_std > 1e-8:
        global_scale = ret_std / cur_std
        if skew_a > 0.2:
            pos_tail_mask = z_scaled > ret_mu
            neg_tail_mask = ~pos_tail_mask
            global_scale_pos = float(np.clip(global_scale, 0.7, 1.05))
            global_scale_neg = float(np.clip(global_scale, 0.7, 1.3))
            z_scaled_new = z_scaled.copy()
            z_scaled_new[pos_tail_mask] = (
                ret_mu + (z_scaled[pos_tail_mask] - ret_mu) * global_scale_pos
            )
            z_scaled_new[neg_tail_mask] = (
                ret_mu + (z_scaled[neg_tail_mask] - ret_mu) * global_scale_neg
            )
            z_scaled = z_scaled_new
        elif skew_a < -0.2:
            neg_tail_mask = z_scaled < ret_mu
            pos_tail_mask = ~neg_tail_mask
            global_scale_neg = float(np.clip(global_scale, 0.7, 1.05))
            global_scale_pos = float(np.clip(global_scale, 0.7, 1.3))
            z_scaled_new = z_scaled.copy()
            z_scaled_new[neg_tail_mask] = (
                ret_mu + (z_scaled[neg_tail_mask] - ret_mu) * global_scale_neg
            )
            z_scaled_new[pos_tail_mask] = (
                ret_mu + (z_scaled[pos_tail_mask] - ret_mu) * global_scale_pos
            )
            z_scaled = z_scaled_new
        else:
            global_scale = float(np.clip(global_scale, 0.7, 1.3))
            z_scaled = ret_mu + (z_scaled - ret_mu) * global_scale

    # ------------------------------------------------------------------
    # Fix-C v30: skew 符號保護 + 幅度限制
    # ------------------------------------------------------------------
    if abs(skew_a) > 0.2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample_skew = float(stats.skew(z_scaled))

        target_sign = int(np.sign(skew_a))
        actual_sign = int(np.sign(sample_skew))
        if actual_sign != 0 and actual_sign != target_sign:
            z_scaled = ret_mu - (z_scaled - ret_mu)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sample_skew = float(stats.skew(z_scaled))

        target_skew_mag = min(abs(skew_a), 1.5)
        if abs(sample_skew) > target_skew_mag * 1.5 and abs(sample_skew) > 0.1:
            damp = target_skew_mag * 1.5 / abs(sample_skew)
            damp = float(np.clip(damp, 0.3, 1.0))
            body_mask_c = np.abs(z_scaled - ret_mu) <= tail_threshold
            z_scaled[body_mask_c] = (
                ret_mu + (z_scaled[body_mask_c] - ret_mu) * damp
            )

    # ------------------------------------------------------------------
    log_rets = z_scaled.copy()

    # jump 注入
    if jump_freq > 0 and n_bars > 0:
        jump_mask  = rng.uniform(size=n_bars) < jump_freq
        jump_sizes = rng.normal(0.0, jump_std, size=n_bars)
        log_rets   = log_rets + np.where(jump_mask, jump_sizes, 0.0)

    # ------------------------------------------------------------------
    # Fix-D v31: jump 後 final global std 收斂
    # Fix-O v36: scale_max = clip(1.0 + (target_ek-3)*0.02, 1.0, 1.4)
    #   不再因 skew 符號而鎖死上限，target_ek 高時任何方向都能讓尾部呼吸
    # ------------------------------------------------------------------
    final_std = float(np.std(log_rets - ret_mu))
    if final_std > 1e-8:
        final_scale = ret_std / final_std
        scale_max = float(np.clip(1.0 + (target_ek - 3.0) * 0.02, 1.0, 1.4))
        final_scale = float(np.clip(final_scale, 0.5, scale_max))
        log_rets = ret_mu + (log_rets - ret_mu) * final_scale

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
# 3. ROLLING FIT -> GENERATE  (v33: 新增 dtw_dist / path_corr)
# ---------------------------------------------------------------------------

def rolling_fit_generate(
    df_real:             pd.DataFrame,
    lookback:            int = 60,
    step:                int = 20,
    n_forward:           int | None = None,
    seed:                int = 42,
    real_anchor_weight:  float = 0.3,
    verbose:             bool = True,
    use_adaptive:        bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    if n_forward is None:
        n_forward = step

    full_rets = _log_returns(df_real["Close"].values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        global_ek = float(stats.kurtosis(full_rets))
    ek_floor = float(np.clip(global_ek, 3.0, 20.0))

    predictor = None
    if use_adaptive:
        try:
            from sim.adaptive_params import OnlineRidgePredictor, extract_features
            predictor = OnlineRidgePredictor(min_train=10, max_blend=0.5,
                                             alpha=1.0, verbose=verbose)
            if verbose:
                print("[adapt] OnlineRidgePredictor enabled (min_train=10, max_blend=0.50)")
        except ImportError:
            if verbose:
                print("[adapt] sklearn not found, adaptive mode disabled")

    n_total    = len(df_real)
    sim_chunks: list[pd.DataFrame] = []
    param_log:  list[dict]         = []
    past_params: list[StatParams]  = []
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
        params    = fit(df_window, apply_trend_bias=True, ek_global_floor=ek_floor)

        if predictor is not None:
            past_params.append(params)
            feats = extract_features(df_real, fit_end, past_params)
            predictor.observe(feats, params)
            if predictor.ready:
                params = predictor.predict_and_blend(feats, params)

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

        real_closes_fwd = real_fwd["Close"].values
        sim_closes_fwd  = df_chunk["Close"].values
        if len(real_closes_fwd) >= 3 and len(sim_closes_fwd) >= 3:
            r_path = np.log(np.maximum(real_closes_fwd, 1e-10)
                            / max(real_closes_fwd[0], 1e-10))
            s_path = np.log(np.maximum(sim_closes_fwd, 1e-10)
                            / max(sim_closes_fwd[0], 1e-10))
            dtw_dist  = round(_dtw_distance(r_path, s_path), 6)
            path_corr = round(_path_corr(real_closes_fwd, sim_closes_fwd), 4)
        else:
            dtw_dist  = float("nan")
            path_corr = float("nan")

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
            "drift_corr":  round(drift_corr, 6),
            "loss":        round(loss, 4),
            "dtw_dist":    dtw_dist,
            "path_corr":   path_corr,
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
                f"tgt_ek={params['target_ek']:.2f}  "
                f"loss={loss:.4f}  "
                f"dtw={dtw_dist:.4f}  pcorr={path_corr:+.3f}"
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

    dtw_vals   = [e["dtw_dist"]  for e in param_log if isinstance(e.get("dtw_dist"), float)
                  and not np.isnan(e["dtw_dist"])]
    pcorr_vals = [e["path_corr"] for e in param_log if isinstance(e.get("path_corr"), float)
                  and not np.isnan(e["path_corr"])]

    if dtw_vals:
        param_log.append({
            "_summary": True,
            "dtw_mean":    round(float(np.mean(dtw_vals)), 6),
            "dtw_median":  round(float(np.median(dtw_vals)), 6),
            "pcorr_mean":  round(float(np.mean(pcorr_vals)), 4) if pcorr_vals else float("nan"),
            "pcorr_median":round(float(np.median(pcorr_vals)), 4) if pcorr_vals else float("nan"),
        })
        if verbose:
            print(
                f"\n[similarity] DTW  mean={np.mean(dtw_vals):.4f}  "
                f"median={np.median(dtw_vals):.4f}  "
                f"(越小越好)"
            )
            if pcorr_vals:
                print(
                    f"[similarity] path_corr  mean={np.mean(pcorr_vals):+.3f}  "
                    f"median={np.median(pcorr_vals):+.3f}  "
                    f"(越接近 +1 越好)"
                )

    return pd.concat(sim_chunks, ignore_index=True), param_log
