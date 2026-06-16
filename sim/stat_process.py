"""
stat_process.py  v45
====================
純統計過程模型，完全不使用 agent。

v45: 三項修正，解決 v44 的 std+50%、skew 反轉、kurtosis+67% 問題
  1. final exact rescale
     - hard-clip 之後強制 (z - mean) / std * ret_std + ret_mu
     - 直接保證輸出 std == ret_std、mean == ret_mu，不受任何前層影響
  2. skew-aware chi2 amp
     - 正尾 amp factor = base * (1 + clamp(skew_a/10, -0.5, 0.5))
     - 負尾 amp factor = base * (1 - clamp(skew_a/10, -0.5, 0.5))
     - skew_a > 0 → 正尾放大 > 負尾，恢復右偏
  3. jump ↔ chi2 互斥
     - target_ek < 5：完全跳過 chi2 amp
     - target_ek >= 5 且做 chi2 amp：jump_freq_eff *= 0.25

v44: 修復 body-only renorm tail explosion + ±8σ hard-clip
v43: body-only renorm（有 mean_shift bug，已修正於 v44）
v42: hard renorm patch
v41b: 修復 OnlineRidgePredictor._ridge_predict 括號 typo
v41 patch: target_ek clip / HIGH_EK_THRESH / chi2 amp 上限
v40: AdaptiveCalibrator 整合
v1-v39: 見舊 docstring
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, TypedDict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from sim.metrics import hurst_exponent

if TYPE_CHECKING:
    from sim.calibrator import AdaptiveCalibrator


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


def _wilder_atr(hi, lo, cl, period=14):
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


def _fit_df_scan(log_rets):
    mu    = float(np.mean(log_rets))
    sigma = float(np.std(log_rets, ddof=1))
    def neg_ll(df):
        return -np.sum(stats.t.logpdf(log_rets, df=df, loc=mu, scale=sigma))
    return float(minimize_scalar(neg_ll, bounds=(2.1, 30.0), method="bounded").x)


def _fit_skewnorm(log_rets):
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


def _fit_wick_lambda(df_ohlc):
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


def _fit_jump_params(log_rets, ret_std):
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


def _fit_vol_persistence(log_rets, ret_mu):
    resid    = log_rets - ret_mu
    resid_sq = resid ** 2
    if len(resid_sq) < 4:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = float(np.corrcoef(resid_sq[:-1], resid_sq[1:])[0, 1])
    return float(np.clip(corr if np.isfinite(corr) else 0.0, 0.0, 0.85))


def _fit_acf_lag1(log_rets):
    if len(log_rets) < 4:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = float(np.corrcoef(log_rets[:-1], log_rets[1:])[0, 1])
    return float(np.clip(corr if np.isfinite(corr) else 0.0, -0.5, 0.5))


def _ar1_hurst_rho(h):
    return float(np.clip(2 ** (2 * h - 1) - 1, -0.95, 0.95))


def _nig_params_from_moments(std, skew, kurt_excess):
    if kurt_excess < 0.5:
        return None
    try:
        a_est = float(np.sqrt(3.0 / max(kurt_excess, 0.1)))
        a_est = float(np.clip(a_est, 0.15, 5.0))
        b_est = float(skew * a_est / 3.0)
        b_est = float(np.clip(b_est, -0.95 * a_est, 0.95 * a_est))
        return a_est, b_est
    except Exception:
        return None


def _dtw_distance(s, t):
    n, m = len(s), len(t)
    if n == 0 or m == 0:
        return float("nan")
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s[i - 1] - t[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m] / max(n, m))


def _path_corr(real_closes, sim_closes):
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

_TARGET_EK_MAX = 30.0


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
    target_ek = float(max(np.clip(ek, 0.5, _TARGET_EK_MAX), ek_global_floor))

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
# 2. GENERATE  (v45)
#
# 修正摘要：
#   Fix-1  final exact rescale
#          hard-clip 後一次 (z-mean)/std*ret_std+ret_mu，保證 std==ret_std
#   Fix-2  skew-aware chi2 amp
#          正尾/負尾用不同的 amp factor，skew_a > 0 → 正尾 > 負尾 → 右偏
#   Fix-3  jump ↔ chi2 互斥
#          target_ek < 5: 不做 chi2 amp
#          target_ek >= 5 且做 chi2: jump_freq_eff *= 0.25
# ---------------------------------------------------------------------------

_AR1_WARMUP     = 50
_HIGH_EK_THRESH = 12.0
_CHI2_AMP_MAX   = 2.5
_TAIL_HARD_CLIP = 8.0


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

    # Fix-3: jump ↔ chi2 互斥 — 決定 chi2 是否啟用，並相應縮減 jump_freq
    use_chi2_amp   = target_ek >= 5.0
    jump_freq_eff  = jump_freq * 0.25 if use_chi2_amp else jump_freq

    HIGH_EK_THRESH = _HIGH_EK_THRESH

    # --- 基礎分佈採樣 ---
    if target_ek > HIGH_EK_THRESH:
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
        t_ek = 6.0 / max(df_t - 4.0, 0.1)
        p_t  = float(np.clip(target_ek / (t_ek + 1e-8), 0.10, 0.92))
        mask       = rng.uniform(size=total) < p_t
        sn_samples = stats.skewnorm.rvs(a=skew_a, loc=0, scale=1,
                                        size=total, random_state=rng)
        t_samples  = stats.t.rvs(df=max(df_t, 2.01), loc=0, scale=1,
                                  size=total, random_state=rng)
        z_raw = np.where(mask, t_samples, sn_samples)

    # --- AR1 Hurst ---
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

    # --- ACF lag1 ---
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

    # --- GJR-GARCH ---
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
            vol_scale = float(np.clip(
                np.sqrt(h_t[i]) / (ret_std + 1e-10), 0.3, 2.5
            ))
            z_gjr[i] = z_adj[i] * vol_scale
    else:
        z_gjr = z_adj

    # --- body-only renorm（v44 修正版） ---
    z_mean = float(np.mean(z_gjr))
    z_std  = float(np.std(z_gjr))
    if z_std > 1e-10:
        body_mask = np.abs(z_gjr - z_mean) < 2.0 * z_std
        z_scaled  = z_gjr.copy()

        if np.any(body_mask):
            body_mean = float(np.mean(z_gjr[body_mask]))
            body_std  = float(np.std(z_gjr[body_mask]))
            if body_std > 1e-10:
                z_scaled[body_mask] = (
                    (z_gjr[body_mask] - body_mean) / body_std * ret_std + ret_mu
                )
            else:
                z_scaled[body_mask] = ret_mu

        if np.any(~body_mask):
            tail_scale = ret_std / z_std
            z_scaled[~body_mask] = (
                (z_gjr[~body_mask] - z_mean) * tail_scale + ret_mu
            )

        cur_mean = float(np.mean(z_scaled))
        z_scaled += (ret_mu - cur_mean)
    else:
        z_scaled = np.full(n_bars, ret_mu)

    # --- Fix-2: skew-aware chi2 tail amp ---
    # Fix-3: target_ek < 5 時完全跳過 chi2 amp
    if use_chi2_amp and target_ek <= HIGH_EK_THRESH:
        tail_threshold = 1.2 * ret_std
        tail_mask      = np.abs(z_scaled - ret_mu) > tail_threshold

        nu_raw   = 3.0 / max(df_t - 4.0, 0.1)
        nu_boost = float(np.clip(nu_raw, 0.5, 3.0))

        # skew_a 決定正尾/負尾的放大比例
        # skew_a > 0 → 正尾 factor 大 → 右偏；反之左偏
        skew_factor = float(np.clip(skew_a / 10.0, -0.5, 0.5))
        pos_tail_boost = 1.0 + skew_factor   # skew_a=+5 → 1.5x base
        neg_tail_boost = 1.0 - skew_factor   # skew_a=+5 → 0.5x base

        if np.any(tail_mask):
            chi2_raw  = rng.chisquare(df=max(df_t, 3.0), size=int(np.sum(tail_mask)))
            chi2_norm = chi2_raw / max(df_t, 3.0)
            extreme_mask_local = chi2_norm > 2.5
            chi2_norm = np.where(
                extreme_mask_local,
                np.clip(chi2_norm, 0.5, 5.0),
                np.clip(chi2_norm, 0.5, 3.5),
            )
            base_amp   = np.clip(1.0 + nu_boost * (chi2_norm - 1.0), 0.5, _CHI2_AMP_MAX)
            sign_mask  = np.sign(z_scaled[tail_mask] - ret_mu)

            # 正尾/負尾分別套用不同 boost
            directional_boost = np.where(sign_mask > 0, pos_tail_boost, neg_tail_boost)
            amp = np.clip(1.0 + (base_amp - 1.0) * directional_boost, 0.5, _CHI2_AMP_MAX)

            z_scaled[tail_mask] = (
                ret_mu + sign_mask * np.abs(z_scaled[tail_mask] - ret_mu) * amp
            )

    # --- jump（Fix-3: 使用縮減後的 jump_freq_eff）---
    if jump_freq_eff > 0 and n_bars > 0:
        n_jumps = int(rng.binomial(n_bars, jump_freq_eff))
        if n_jumps > 0:
            jump_idx   = rng.choice(n_bars, size=n_jumps, replace=False)
            jump_sizes = rng.normal(0.0, jump_std, size=n_jumps)
            z_scaled[jump_idx] += jump_sizes

    # --- tail hard-clip ±8σ ---
    clip_lo  = ret_mu - _TAIL_HARD_CLIP * ret_std
    clip_hi  = ret_mu + _TAIL_HARD_CLIP * ret_std
    z_clipped = np.clip(z_scaled, clip_lo, clip_hi)

    # --- Fix-1: final exact rescale — 保證 std == ret_std、mean == ret_mu ---
    z_cur_std = float(np.std(z_clipped))
    if z_cur_std > 1e-10:
        z_final_out = (z_clipped - float(np.mean(z_clipped))) / z_cur_std * ret_std + ret_mu
    else:
        z_final_out = np.full(n_bars, ret_mu)

    # --- 價格序列 ---
    prices    = np.empty(n_bars + 1)
    prices[0] = start_price
    for i in range(n_bars):
        prices[i + 1] = prices[i] * np.exp(z_final_out[i])

    opens  = prices[:-1].copy()
    closes = prices[1:].copy()
    atr_adj = max(atr_mean, 1e-4)
    upper_w = rng.exponential(scale=wick_lam * atr_adj, size=n_bars)
    lower_w = rng.exponential(scale=wick_lam * atr_adj, size=n_bars)
    highs   = np.maximum(opens, closes) + upper_w
    lows    = np.maximum(np.minimum(opens, closes) - lower_w, 1e-6)

    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes})


# ---------------------------------------------------------------------------
# 3. OnlineRidgePredictor
# ---------------------------------------------------------------------------

class OnlineRidgePredictor:
    def __init__(self, min_train=10, max_blend=0.50):
        self.min_train = min_train
        self.max_blend = max_blend
        self._X: list = []
        self._y_std:   list = []
        self._y_skew:  list = []
        self._y_hurst: list = []
        self._y_ek:    list = []

    def record(self, params, realised_std, realised_skew, realised_hurst, realised_ek):
        feat = [
            params["ret_std"], params["ret_skew_a"],
            params["hurst_target"], params["target_ek"],
            params["ret_df"], params["vol_persistence"],
        ]
        self._X.append(feat)
        self._y_std.append(realised_std)
        self._y_skew.append(realised_skew)
        self._y_hurst.append(realised_hurst)
        self._y_ek.append(realised_ek)

    def _ridge_predict(self, X, y, x_new, alpha=1.0):
        n, d = X.shape
        XtX = X.T @ X + alpha * np.eye(d)
        try:
            w = np.linalg.solve(XtX, X.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
        return float(x_new @ w)

    def predict_correction(self, params, window_idx):
        n = len(self._X)
        if n < self.min_train:
            return params
        blend = float(np.clip(
            self.max_blend * (n - self.min_train) / max(self.min_train, 1),
            0.0, self.max_blend
        ))
        X = np.array(self._X, dtype=float)
        x_new = np.array([
            params["ret_std"], params["ret_skew_a"],
            params["hurst_target"], params["target_ek"],
            params["ret_df"], params["vol_persistence"],
        ], dtype=float)
        new_std   = float(np.clip((1-blend)*params["ret_std"]      + blend*self._ridge_predict(X, np.array(self._y_std),   x_new), params["ret_std"]*0.3, params["ret_std"]*3.0))
        new_skew  = float(np.clip((1-blend)*params["ret_skew_a"]   + blend*self._ridge_predict(X, np.array(self._y_skew),  x_new), -10.0, 10.0))
        new_hurst = float(np.clip((1-blend)*params["hurst_target"] + blend*self._ridge_predict(X, np.array(self._y_hurst), x_new), 0.3, 0.69))
        new_ek    = float(np.clip((1-blend)*params["target_ek"]    + blend*self._ridge_predict(X, np.array(self._y_ek),    x_new), 1.0, _TARGET_EK_MAX))
        corrected = dict(params)
        corrected.update({"ret_std": new_std, "ret_skew_a": new_skew,
                          "hurst_target": new_hurst, "target_ek": new_ek})
        return StatParams(**corrected)


# ---------------------------------------------------------------------------
# 4. Rolling fit-generate
# ---------------------------------------------------------------------------

def rolling_fit_generate(
    df_real:    pd.DataFrame,
    lookback:   int  = 60,
    step:       int  = 20,
    seed:       int  = 42,
    verbose:    bool = False,
    use_adapt:  bool = True,
    calibrator: "Optional[AdaptiveCalibrator]" = None,
) -> tuple[pd.DataFrame, list[dict]]:
    n      = len(df_real)
    pos    = 0
    result_chunks: list[pd.DataFrame] = []
    param_log: list[dict] = []
    rng    = np.random.default_rng(seed)
    window_idx = 0

    predictor = OnlineRidgePredictor(min_train=10, max_blend=0.50) if use_adapt else None

    all_dtw:   list[float] = []
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

        calib_action = None
        if calibrator is not None:
            from sim.calibrator import AdaptiveCalibrator
            ctx = AdaptiveCalibrator.build_context(params)
            calib_action = calibrator.predict(ctx)
            params_dict  = calib_action.apply(dict(params))
            params       = StatParams(**params_dict)
        elif predictor is not None:
            params = predictor.predict_correction(params, window_idx)

        start_price = float(df_real["Open"].iloc[fwd_start])
        sim_seed    = int(rng.integers(0, 2**31))
        df_sim = generate(params=params, n_bars=fwd_bars,
                          start_price=start_price, seed=sim_seed)

        sim_rets  = np.diff(np.log(np.maximum(df_sim["Close"].values,  1e-10)))
        real_rets = np.diff(np.log(np.maximum(df_fwd["Close"].values,  1e-10)))

        real_std_w  = float(np.std(real_rets)) + 1e-10
        std_err_pct = abs(float(np.std(sim_rets)) / real_std_w - 1.0) if len(sim_rets) > 1 else float("nan")
        kurt_err    = abs(float(stats.kurtosis(sim_rets)) - float(stats.kurtosis(real_rets))) if len(sim_rets) > 3 else float("nan")
        hurst_err   = abs(float(hurst_exponent(sim_rets)) - float(hurst_exponent(real_rets))) if len(sim_rets) > 10 else float("nan")
        min_len     = min(len(real_rets), len(sim_rets))
        dir_hit     = float(np.mean(np.sign(real_rets[:min_len]) == np.sign(sim_rets[:min_len]))) if min_len > 0 else float("nan")

        if len(sim_rets) > 1 and len(real_rets) > 1:
            loss = float(
                abs(np.std(sim_rets) - np.std(real_rets)) / real_std_w
                + abs(np.mean(sim_rets) - np.mean(real_rets)) / real_std_w
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

        if calibrator is not None and calib_action is not None:
            if all(np.isfinite(v) for v in [std_err_pct, kurt_err, hurst_err, dir_hit]):
                calibrator.record(
                    context     = ctx,
                    action      = calib_action,
                    std_err_pct = std_err_pct,
                    kurt_err    = kurt_err,
                    hurst_err   = hurst_err,
                    dir_hit     = dir_hit,
                )

        if predictor is not None and len(real_rets) > 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictor.record(
                    params,
                    float(np.std(real_rets)),
                    float(stats.skew(real_rets)),
                    float(hurst_exponent(real_rets)) if len(real_rets) > 10 else params["hurst_target"],
                    float(stats.kurtosis(real_rets)),
                )

        real_c = float(df_fwd["Close"].iloc[-1])
        sim_c  = float(df_sim["Close"].iloc[-1])
        c_err  = (sim_c - real_c) / (real_c + 1e-10)

        if verbose:
            dtw_str   = f"{dtw_val:.4f}"   if np.isfinite(dtw_val)   else "  nan"
            pcorr_str = f"{pcorr_val:+.3f}" if np.isfinite(pcorr_val) else "  nan"
            calib_str = f"  [calib n={calibrator.n_experiences}]" if calibrator else ""
            print(
                f"[stat] w{window_idx:3d}"
                f"  std={params['ret_std']:.4f}  ek={params['target_ek']:.2f}"
                f"  loss={loss:.4f}  dtw={dtw_str}  pcorr={pcorr_str}"
                f"  dir={dir_hit:.3f}{calib_str}"
            )

        param_log.append({
            "window":    window_idx,
            "fit_range": [fit_start, fit_end],
            "fwd_bars":  [fwd_start, fwd_end],
            **{k: float(v) for k, v in params.items()},
            "loss":      loss      if np.isfinite(loss)      else None,
            "dtw":       dtw_val   if np.isfinite(dtw_val)   else None,
            "path_corr": pcorr_val if np.isfinite(pcorr_val) else None,
            "c_err":     float(c_err),
        })

        result_chunks.append(df_sim)
        pos += step

    df_result = pd.concat(result_chunks, ignore_index=True)

    if all_dtw or all_pcorr:
        dtw_mean   = float(np.mean(all_dtw))     if all_dtw   else float("nan")
        dtw_median = float(np.median(all_dtw))   if all_dtw   else float("nan")
        pc_mean    = float(np.mean(all_pcorr))   if all_pcorr else float("nan")
        pc_median  = float(np.median(all_pcorr)) if all_pcorr else float("nan")
        print(f"\n[similarity] DTW  mean={dtw_mean:.4f}  median={dtw_median:.4f}  (越小越好)")
        print(f"[similarity] path_corr  mean={pc_mean:+.3f}  median={pc_median:+.3f}  (越接近 +1 越好)")

    param_log.append({
        "_summary":   True,
        "n_windows":  window_idx,
        "dtw_mean":   float(np.mean(all_dtw))   if all_dtw   else None,
        "dtw_median": float(np.median(all_dtw)) if all_dtw   else None,
        "pcorr_mean": float(np.mean(all_pcorr)) if all_pcorr else None,
    })

    return df_result, param_log
