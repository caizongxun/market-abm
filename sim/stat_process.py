"""
stat_process.py  v38
====================
純統計過程模型，完全不使用 agent。

v38 新增（v37 骨架不動）：
------------------------------------------------------
Fix-38-P1  NIG 穩定性保護 + 超高 ek 截斷 t-mixture
  問題：target_ek > 20 時 a_nig = sqrt(3/ek) < 0.15，NIG 退化成柯西，
        final_scale 壓不住 → kurt_err 爆炸（GOOGL=437）。
  修法：
    (a) a_nig 強制 >= 0.15（最胖尾下限保護）。
    (b) target_ek > 20 時改走「高 ek 截斷 t-mixture」路徑：
        df_t_eff = max(df_t, 2.5) → 確保真實重尾
        nu_boost 上限從 3.0 → 5.0（允許更激進尾部放大）
        p_t 上限從 0.92 → 0.98（幾乎全走 t-dist）
  預期：kurt_err 中位數 21 → 8

Fix-38-P2  GJR vol_scale 收緊 + scale_max 壓制
  問題：vol_scale clip(0.3, 3.5) 在 GLD/TLT/GOOGL 把 std 放大超標。
  修法：
    (a) vol_scale clip 收緊至 (0.3, 2.5)。
    (b) scale_max = min(1.0 + (ek-3)*0.015, 1.2)（原 0.02 係數改 0.015，上限 1.4→1.2）。
  預期：std_err% 中位數 15 → 10

v1-v38 修正歷程
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
  Fix-37  : Fix-P GJR-GARCH 槓桿效應（下跌期波動率放大）
             Fix-Q NIG 尾部採樣（target_ek>6 時替換 t-mixture）
  Fix-38  : Fix-38-P1 NIG a_nig >= 0.15 下限 + target_ek > 20 截斷 t-mixture
             Fix-38-P2 GJR vol_scale clip(0.3,2.5) + scale_max 係數 0.02→0.015 上限 1.4→1.2
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
# Fix-Q v37 / Fix-38-P1: NIG 矩匹配工具函數
# ---------------------------------------------------------------------------

def _nig_params_from_moments(
    std: float, skew: float, kurt_excess: float
) -> tuple[float, float] | None:
    """
    從超額峰度和偏度推算 NIG(a, b) 參數（scipy norminvgauss 形式）。
    Fix-38-P1: a_nig 強制 >= 0.15（防止柯西退化）。
    target_ek > 20 時由呼叫方改走截斷 t-mixture，不進此函數。
    """
    if kurt_excess < 0.5:
        return None
    try:
        a_est = float(np.sqrt(3.0 / max(kurt_excess, 0.1)))
        # Fix-38-P1: 下限保護 0.15，防止 a→0 退化成柯西分佈
        a_est = float(np.clip(a_est, 0.15, 5.0))
        b_est = float(skew * a_est / 3.0)
        b_est = float(np.clip(b_est, -0.95 * a_est, 0.95 * a_est))
        return a_est, b_est
    except Exception:
        return None


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
# 2. GENERATE  (v38: Fix-38-P1 NIG穩定 + Fix-38-P2 GJR收緊)
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

    total = n_bars + _AR1_WARMUP

    # ------------------------------------------------------------------
    # Fix-38-P1: 根據 target_ek 選擇基礎分佈
    #   target_ek <= 6  : t-mixture（原路徑）
    #   6 < target_ek <= 20 : NIG（矩匹配，a_nig >= 0.15）
    #   target_ek > 20  : 高 ek 截斷 t-mixture（p_t→0.98, nu_boost↑）
    # ------------------------------------------------------------------
    HIGH_EK_THRESH = 20.0

    if target_ek > HIGH_EK_THRESH:
        # 截斷 t-mixture：幾乎純 t，df 強制收緊到真實重尾範圍
        use_nig    = False
        df_t_eff   = float(np.clip(df_t, 2.5, 8.0))
        t_ek_eff   = 6.0 / max(df_t_eff - 4.0, 0.1)
        p_t        = float(np.clip(target_ek / (t_ek_eff + 1e-8), 0.10, 0.98))
        mask       = rng.uniform(size=total) < p_t
        sn_samples = stats.skewnorm.rvs(a=skew_a, loc=0, scale=1,
                                        size=total, random_state=rng)
        t_samples  = stats.t.rvs(df=df_t_eff, loc=0, scale=1,
                                  size=total, random_state=rng)
        z_raw = np.where(mask, t_samples, sn_samples)
    elif target_ek > 6.0:
        # NIG 路徑（Fix-Q + Fix-38-P1 保護）
        nig_ab = _nig_params_from_moments(ret_std, skew_a, target_ek)
        use_nig = nig_ab is not None
        if use_nig:
            a_nig, b_nig = nig_ab
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    z_raw = stats.norminvgauss.rvs(
                        a=a_nig, b=b_nig, loc=0, scale=1,
                        size=total, random_state=int(rng.integers(0, 2**31))
                    )
                except Exception:
                    use_nig = False
        if not use_nig:
            t_ek = 6.0 / max(df_t - 4.0, 0.1)
            p_t  = float(np.clip(target_ek / (t_ek + 1e-8), 0.10, 0.92))
            mask       = rng.uniform(size=total) < p_t
            sn_samples = stats.skewnorm.rvs(a=skew_a, loc=0, scale=1,
                                            size=total, random_state=rng)
            t_samples  = stats.t.rvs(df=max(df_t, 2.01), loc=0, scale=1,
                                      size=total, random_state=rng)
            z_raw = np.where(mask, t_samples, sn_samples)
    else:
        # 原 t-mixture 路徑（target_ek <= 6）
        use_nig = False
        t_ek = 6.0 / max(df_t - 4.0, 0.1)
        p_t  = float(np.clip(target_ek / (t_ek + 1e-8), 0.10, 0.92))
        mask       = rng.uniform(size=total) < p_t
        sn_samples = stats.skewnorm.rvs(a=skew_a, loc=0, scale=1,
                                        size=total, random_state=rng)
        t_samples  = stats.t.rvs(df=max(df_t, 2.01), loc=0, scale=1,
                                  size=total, random_state=rng)
        z_raw = np.where(mask, t_samples, sn_samples)

    # ------------------------------------------------------------------
    # AR(1) / Hurst 結構
    # ------------------------------------------------------------------
    rho = _ar1_hurst_rho(hurst)
    if abs(rho) > 1e-6:
        innov_sc    = float(np.sqrt(max(1.0 - rho ** 2, 0.0)))
        z_ar_ext    = np.empty(total)
        z_ar_ext[0] = z_raw[0]
        for i in range(1, total):
            z_ar_ext[i] = rho * z_ar_ext[i - 1] + innov_sc * z_raw[i]
        z_ar = z_ar_ext[_AR1_WARMUP:]
    else:
        z_ar = z_raw[_AR1_WARMUP:].copy()

    # ------------------------------------------------------------------
    # ACF(1) 修正
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Fix-P v37 / Fix-38-P2: GJR-GARCH（不對稱槓桿效應）
    # Fix-38-P2: vol_scale clip 收緊至 (0.3, 2.5)（原 3.5）
    # ------------------------------------------------------------------
    alpha = vol_persistence
    if alpha > 0.01 and n_bars > 1:
        gamma    = float(np.clip(alpha * 0.6, 0.0, 0.4))
        beta     = float(np.clip(alpha * 0.7, 0.0, 0.89))
        alpha_u  = alpha * (1.0 - gamma)
        alpha_d  = alpha * (1.0 + gamma)
        base_var = ret_std ** 2
        omega    = base_var * max(1.0 - beta - 0.5*(alpha_u + alpha_d), 0.01)

        h_t        = np.empty(n_bars)
        z_gjr      = np.empty(n_bars)
        h_t[0]     = base_var
        z_gjr[0]   = z_adj[0]

        for i in range(1, n_bars):
            prev_ret = z_gjr[i - 1] * ret_std
            if prev_ret < 0:
                resid_term = alpha_d * prev_ret ** 2
            else:
                resid_term = alpha_u * prev_ret ** 2
            h_t[i]   = omega + beta * h_t[i - 1] + resid_term
            h_t[i]   = max(h_t[i], base_var * 0.01)
            # Fix-38-P2: clip 上限 3.5 → 2.5
            vol_scale = float(np.clip(
                np.sqrt(h_t[i]) / (ret_std + 1e-10), 0.3, 2.5
            ))
            z_gjr[i] = z_adj[i] * vol_scale
    else:
        z_gjr = z_adj

    # ------------------------------------------------------------------
    # 全局 std 收斂（body-only rescale，不碰尾部）
    # ------------------------------------------------------------------
    z_mean = float(np.mean(z_gjr))
    z_std  = float(np.std(z_gjr))
    if z_std > 1e-10:
        z_body   = z_gjr.copy()
        body_idx = np.abs(z_gjr - z_mean) < 2.0 * z_std
        z_body[body_idx] = (
            (z_gjr[body_idx] - z_mean) / z_std * ret_std + ret_mu
        )
        z_body[~body_idx] = z_gjr[~body_idx] - z_mean + ret_mu
        z_scaled = z_body
    else:
        z_scaled = np.full(n_bars, ret_mu)

    # ------------------------------------------------------------------
    # Chi2 尾部放大（Fix-I/L/M 繼承）
    # Fix-38-P1: target_ek > 20 時 nu_boost 上限從 3.0 → 5.0
    # ------------------------------------------------------------------
    tail_threshold = 1.2 * ret_std
    tail_mask      = np.abs(z_scaled - ret_mu) > tail_threshold

    nu_raw   = 3.0 / max(df_t - 4.0, 0.1)
    # Fix-38-P1: 超高 ek 品種允許更激進的尾部放大
    nu_boost_max = 5.0 if target_ek > HIGH_EK_THRESH else 3.0
    nu_boost = float(np.clip(nu_raw, 0.5, nu_boost_max))

    if np.any(tail_mask):
        chi2_raw = rng.chisquare(df=max(df_t, 3.0), size=int(np.sum(tail_mask)))
        chi2_norm = chi2_raw / max(df_t, 3.0)

        extreme_mask_local = chi2_norm > 2.5
        chi2_norm = np.where(
            extreme_mask_local,
            np.clip(chi2_norm, 0.5, 5.0),
            np.clip(chi2_norm, 0.5, 3.5),
        )
        amp = 1.0 + nu_boost * (chi2_norm - 1.0)
        amp = np.clip(amp, 0.5, 4.0)
        sign_mask = np.sign(z_scaled[tail_mask] - ret_mu)
        z_scaled[tail_mask] = (
            ret_mu + sign_mask * np.abs(z_scaled[tail_mask] - ret_mu) * amp
        )

    # ------------------------------------------------------------------
    # Jump diffusion（Merton 跳躍項）
    # ------------------------------------------------------------------
    if jump_freq > 0 and n_bars > 0:
        n_jumps = int(rng.binomial(n_bars, jump_freq))
        if n_jumps > 0:
            jump_idx   = rng.choice(n_bars, size=n_jumps, replace=False)
            jump_sizes = rng.normal(0.0, jump_std, size=n_jumps)
            z_scaled[jump_idx] += jump_sizes

    # ------------------------------------------------------------------
    # Fix-38-P2: scale_max 係數 0.02→0.015，上限 1.4→1.2
    # ------------------------------------------------------------------
    scale_max   = float(np.clip(1.0 + (target_ek - 3.0) * 0.015, 1.0, 1.2))
    final_mean  = float(np.mean(z_scaled))
    final_std   = float(np.std(z_scaled))
    if final_std > 1e-10:
        final_scale = float(np.clip(ret_std / final_std, 0.5, scale_max))
        z_scaled    = (z_scaled - final_mean) * final_scale + ret_mu

    # ------------------------------------------------------------------
    # 價格路徑重建
    # ------------------------------------------------------------------
    log_rets_sim = z_scaled
    prices       = np.empty(n_bars + 1)
    prices[0]    = start_price
    for i in range(n_bars):
        prices[i + 1] = prices[i] * np.exp(log_rets_sim[i])

    opens  = prices[:-1].copy()
    closes = prices[1:].copy()

    atr_adj = max(atr_mean, 1e-4)
    upper_w = rng.exponential(scale=wick_lam * atr_adj, size=n_bars)
    lower_w = rng.exponential(scale=wick_lam * atr_adj, size=n_bars)
    highs   = np.maximum(opens, closes) + upper_w
    lows    = np.minimum(opens, closes) - lower_w
    lows    = np.maximum(lows, 1e-6)

    return pd.DataFrame({
        "Open":  opens,
        "High":  highs,
        "Low":   lows,
        "Close": closes,
    })


# ---------------------------------------------------------------------------
# 3. OnlineRidgePredictor  (自適應參數修正，v25+)
# ---------------------------------------------------------------------------

class OnlineRidgePredictor:
    """
    使用已完成視窗的 (fitted_params -> realised_moment) 配對，
    以 Ridge 回歸修正下一個視窗的參數。
    目前修正：ret_std、ret_skew_a（skew）、hurst_target、target_ek。
    """
    def __init__(self, min_train: int = 10, max_blend: float = 0.50):
        self.min_train = min_train
        self.max_blend = max_blend
        self._X: list[list[float]] = []
        self._y_std:   list[float] = []
        self._y_skew:  list[float] = []
        self._y_hurst: list[float] = []
        self._y_ek:    list[float] = []

    def record(self, params: StatParams, realised_std: float,
               realised_skew: float, realised_hurst: float,
               realised_ek: float) -> None:
        feat = [
            params["ret_std"],
            params["ret_skew_a"],
            params["hurst_target"],
            params["target_ek"],
            params["ret_df"],
            params["vol_persistence"],
        ]
        self._X.append(feat)
        self._y_std.append(realised_std)
        self._y_skew.append(realised_skew)
        self._y_hurst.append(realised_hurst)
        self._y_ek.append(realised_ek)

    def _ridge_predict(self, X: np.ndarray, y: np.ndarray,
                       x_new: np.ndarray, alpha: float = 1.0) -> float:
        n, d = X.shape
        XtX = X.T @ X + alpha * np.eye(d)
        Xty = X.T @ y
        try:
            w = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        pred = float(x_new @ w)
        return pred

    def predict_correction(
        self, params: StatParams, window_idx: int
    ) -> StatParams:
        n = len(self._X)
        if n < self.min_train:
            return params

        blend = float(np.clip(
            self.max_blend * (n - self.min_train) / max(self.min_train, 1),
            0.0, self.max_blend
        ))

        X     = np.array(self._X, dtype=float)
        x_new = np.array([
            params["ret_std"],
            params["ret_skew_a"],
            params["hurst_target"],
            params["target_ek"],
            params["ret_df"],
            params["vol_persistence"],
        ], dtype=float)

        pred_std   = self._ridge_predict(X, np.array(self._y_std),   x_new)
        pred_skew  = self._ridge_predict(X, np.array(self._y_skew),  x_new)
        pred_hurst = self._ridge_predict(X, np.array(self._y_hurst), x_new)
        pred_ek    = self._ridge_predict(X, np.array(self._y_ek),    x_new)

        new_std   = float(np.clip(
            (1 - blend) * params["ret_std"]      + blend * pred_std,
            params["ret_std"] * 0.3, params["ret_std"] * 3.0
        ))
        new_skew  = float(np.clip(
            (1 - blend) * params["ret_skew_a"]   + blend * pred_skew,
            -10.0, 10.0
        ))
        new_hurst = float(np.clip(
            (1 - blend) * params["hurst_target"] + blend * pred_hurst,
            0.3, 0.69
        ))
        new_ek    = float(np.clip(
            (1 - blend) * params["target_ek"]    + blend * pred_ek,
            1.0, 30.0
        ))

        corrected = dict(params)
        corrected["ret_std"]      = new_std
        corrected["ret_skew_a"]   = new_skew
        corrected["hurst_target"] = new_hurst
        corrected["target_ek"]    = new_ek
        return StatParams(**corrected)


# ---------------------------------------------------------------------------
# 4. Rolling fit-generate  (滾動主函數)
# ---------------------------------------------------------------------------

def rolling_fit_generate(
    df_real:   pd.DataFrame,
    lookback:  int  = 60,
    step:      int  = 20,
    seed:      int  = 42,
    verbose:   bool = False,
    use_adapt: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:

    n      = len(df_real)
    pos    = 0
    result_chunks: list[pd.DataFrame] = []
    param_log: list[dict] = []
    rng    = np.random.default_rng(seed)
    window_idx = 0

    predictor = OnlineRidgePredictor(min_train=10, max_blend=0.50) if use_adapt else None

    all_dtw:  list[float] = []
    all_pcorr: list[float] = []

    while pos + lookback < n:
        window_idx += 1
        fit_start = pos
        fit_end   = pos + lookback
        fwd_start = fit_end
        fwd_end   = min(fwd_start + step, n)
        fwd_bars  = fwd_end - fwd_start

        df_fit = df_real.iloc[fit_start:fit_end].copy().reset_index(drop=True)
        df_fwd = df_real.iloc[fwd_start:fwd_end].copy().reset_index(drop=True)

        params = fit(df_fit)

        if predictor is not None:
            params = predictor.predict_correction(params, window_idx)
            if verbose and len(predictor._X) >= predictor.min_train:
                n_hist = len(predictor._X)
                blend_w = float(np.clip(
                    predictor.max_blend * (n_hist - predictor.min_train) / max(predictor.min_train, 1),
                    0.0, predictor.max_blend
                ))
                print(
                    f"         [adapt] blend_w={blend_w:.2f}"
                    f"  std {dict(params)['ret_std']:.4f}->{params['ret_std']:.4f}"
                    f"  skew {dict(params)['ret_skew_a']:.3f}->{params['ret_skew_a']:.3f}"
                    f"  hurst {dict(params)['hurst_target']:.3f}->{params['hurst_target']:.3f}"
                    f"  tgt_ek {dict(params)['target_ek']:.2f}->{params['target_ek']:.2f}"
                    f"  (n={n_hist})"
                )

        start_price = float(df_real["Open"].iloc[fwd_start])
        sim_seed    = int(rng.integers(0, 2**31))

        df_sim = generate(
            params      = params,
            n_bars      = fwd_bars,
            start_price = start_price,
            seed        = sim_seed,
        )

        sim_rets  = np.diff(np.log(np.maximum(df_sim["Close"].values,  1e-10)))
        real_rets = np.diff(np.log(np.maximum(df_fwd["Close"].values,  1e-10)))
        if len(sim_rets) > 1 and len(real_rets) > 1:
            loss = float(
                abs(np.std(sim_rets) - np.std(real_rets)) / (np.std(real_rets) + 1e-10)
                + abs(np.mean(sim_rets) - np.mean(real_rets)) / (np.std(real_rets) + 1e-10)
            )
        else:
            loss = float("nan")

        dtw_val   = _dtw_distance(
            real_rets / (np.std(real_rets) + 1e-10),
            sim_rets  / (np.std(sim_rets)  + 1e-10),
        ) if len(sim_rets) > 2 and len(real_rets) > 2 else float("nan")
        pcorr_val = _path_corr(df_fwd["Close"].values, df_sim["Close"].values)

        if np.isfinite(dtw_val):   all_dtw.append(dtw_val)
        if np.isfinite(pcorr_val): all_pcorr.append(pcorr_val)

        if predictor is not None and len(real_rets) > 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_std   = float(np.std(real_rets))
                r_skew  = float(stats.skew(real_rets))
                r_hurst = float(hurst_exponent(real_rets)) if len(real_rets) > 10 else params["hurst_target"]
                r_ek    = float(stats.kurtosis(real_rets))
            predictor.record(params, r_std, r_skew, r_hurst, r_ek)

        real_o = float(df_fwd["Open"].iloc[0])
        real_h = float(df_fwd["High"].max())
        real_l = float(df_fwd["Low"].min())
        real_c = float(df_fwd["Close"].iloc[-1])
        sim_o  = float(df_sim["Open"].iloc[0])
        sim_h  = float(df_sim["High"].max())
        sim_l  = float(df_sim["Low"].min())
        sim_c  = float(df_sim["Close"].iloc[-1])
        c_err  = (sim_c - real_c) / (real_c + 1e-10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_rets = np.diff(np.log(np.maximum(df_fit["Close"].values, 1e-10)))
            skew_actual  = float(stats.skew(fit_rets))   if len(fit_rets) > 3 else 0.0
            hurst_actual = float(hurst_exponent(fit_rets)) if len(fit_rets) > 10 else 0.5
            wick_actual  = float(np.mean(_wilder_atr(
                df_fit["High"].values, df_fit["Low"].values, df_fit["Close"].values, 14
            )))

        if verbose:
            dtw_str   = f"{dtw_val:.4f}"   if np.isfinite(dtw_val)   else "  nan"
            pcorr_str = f"{pcorr_val:+.3f}" if np.isfinite(pcorr_val) else "  nan"
            print(
                f"[stat] window {window_idx:3d}"
                f"  fit=[{fit_start}:{fit_end}]"
                f"  fwd=[{fwd_start}:{fwd_end}]"
                f"  df={params['ret_df']:.2f}"
                f"  skew_a={params['ret_skew_a']:+.3f}"
                f"  std={params['ret_std']:.4f}"
                f"  hurst={params['hurst_target']:.3f}"
                f"  wick={params['wick_lambda']:.3f}"
                f"  jfreq={params['jump_freq']:.3f}"
                f"  vp={params['vol_persistence']:.3f}"
                f"  acf1={params['acf_lag1']:+.3f}"
                f"  tgt_ek={params['target_ek']:.2f}"
                f"  loss={loss:.4f}"
                f"  dtw={dtw_str}"
                f"  pcorr={pcorr_str}"
            )
            print(
                f"         real OHLC"
                f"  O={real_o:8.2f}  H={real_h:8.2f}  L={real_l:8.2f}  C={real_c:8.2f}"
            )
            print(
                f"         sim  OHLC"
                f"  O={sim_o:8.2f}  H={sim_h:8.2f}  L={sim_l:8.2f}  C={sim_c:8.2f}"
                f"  (C err={c_err:+.1%})"
            )

        param_log.append({
            "window":         window_idx,
            "fit_range":      [fit_start, fit_end],
            "fwd_bars":       [fwd_start, fwd_end],
            **{k: float(v) for k, v in params.items()},
            "loss":           loss if np.isfinite(loss) else None,
            "dtw":            dtw_val if np.isfinite(dtw_val) else None,
            "path_corr":      pcorr_val if np.isfinite(pcorr_val) else None,
            "c_err":          float(c_err),
        })

        result_chunks.append(df_sim)
        pos += step

    df_result = pd.concat(result_chunks, ignore_index=True)

    if all_dtw or all_pcorr:
        dtw_mean   = float(np.mean(all_dtw))   if all_dtw   else float("nan")
        dtw_median = float(np.median(all_dtw)) if all_dtw   else float("nan")
        pc_mean    = float(np.mean(all_pcorr)) if all_pcorr else float("nan")
        pc_median  = float(np.median(all_pcorr)) if all_pcorr else float("nan")
        print(f"\n[similarity] DTW  mean={dtw_mean:.4f}  median={dtw_median:.4f}  (越小越好)")
        print(f"[similarity] path_corr  mean={pc_mean:+.3f}  median={pc_median:+.3f}  (越接近 +1 越好)")

    param_log.append({
        "_summary": True,
        "n_windows": window_idx,
        "dtw_mean":  float(np.mean(all_dtw))   if all_dtw   else None,
        "dtw_median":float(np.median(all_dtw)) if all_dtw   else None,
        "pcorr_mean":float(np.mean(all_pcorr)) if all_pcorr else None,
    })

    return df_result, param_log
