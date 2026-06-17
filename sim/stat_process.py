"""
stat_process.py  v54
====================
純統計過程模型，完全不使用 agent。

v54: Fix four problems observed in v53 output:
       (skew=0.448→-1.111, kurtosis=10.6→6.4, hurst=0.693→0.660)

     Root causes:
       1. _kurtosis_topup replaces smallest-|ret| positions with jumps drawn from
          symmetric normal(skew_bias*jstd, jstd).  With large jstd the negative
          tail dominates, pulling global skew negative regardless of skew_bias sign.
       2. After topup, no skew correction pass → skew -1.11 persists.
       3. topup "replace" mode: out[ix] = jump breaks mean/std silently; additive
          mode out[ix] += jump is more controlled and raises 4th moment faster.
       4. Each window clip skew_raw to its own observed value (can be ±1.2), so
          when windows are concatenated the aggregate skew drifts away from the
          global target.  The AR1 Hurst pass then re-correlates this mixed-sign
          series, degrading hurst accuracy.

     Fixes:
       A. Global skew anchor in rolling_fit_generate:
          Compute global_skew_target from the full df_real at the start.
          Each window's skew_raw is soft-clipped to
          [global_skew_target - 0.5, global_skew_target + 0.5]
          before being passed to generate(). This keeps per-window skew
          from wandering too far from the dataset's true skew direction.

       B. _kurtosis_topup: additive mode (out[ix] += jump_draws)
          Replace → additive: injected jumps ADD to existing returns so std
          and mean grow only marginally while the 4th moment rises sharply.
          Also: n_jumps formula tuned (deficit * len / 20 instead of / 30)
          and topup_frac lowered to 0.50 to trigger earlier.

       C. Global skew correction pass after topup:
          After _kurtosis_topup, apply _skew_post_shift(final_rets_topup,
          target_skew=global_skew_target) to snap aggregate skew back to
          the measured global value. Clamp unchanged at 0.30.

       D. Hurst guard: reduce _skew_post_shift clamp inside generate()
          from 0.30 → 0.20 for per-window calls (only the global pass keeps
          0.30).  Smaller per-window cubic distortion preserves AR1 structure.

     All other stages (nct iterative nc, AR1 Hurst, ACF, GARCH, volume,
     OHLC wick, OnlineRidgePredictor) unchanged from v53.

v53: Fix three residual problems from v52 output:
       (skew=0.448→-0.135, kurtosis=10.6→3.9, topup ineffective)
v52.1: fix SyntaxError in OnlineRidgePredictor.predict_correction (line 743-744)
v52: Fix kurtosis collapse (10.6 → 1.3) and skew collapse (0.45 → 0.09).
v51: Replace t-mixture (v50) with noncentral-t (nct) sampling.
v50: Remove Fleishman, t-mixture + jump-amplify (insufficient ek)
v49.3: fix bracket mismatch
v49.2: fix rolling level drift
v49.1: fix Fleishman mean-drift
v49: Fleishman Power Transform (abandoned)
v48.1: fix bracket mismatch
v48: std inflation, skew sign fixes
v47: Volume fit / generate
v46: GARCH(1,1) + exact rescale
v1-v45: 見舊 docstring
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, TypedDict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar, brentq

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
    vol_log_mean:     float
    vol_log_std:      float
    vol_ret_beta:     float
    vol_ar1:          float
    ret_skew_raw:     float


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
# v53/v54 helpers: nct iterative nc, skew post-shift, kurtosis top-up
# ---------------------------------------------------------------------------

_NCT_DF_FLOOR = 4.05
_NCT_DF_CEIL  = 30.0
_NCT_NC_MAX   = 8.0

# v53 Fix B: pipeline degrades kurtosis ~55%; pre-amplify target_ek by this factor
_EK_OVERSAMPLE = 1.8


def _df_from_ek(target_ek: float) -> float:
    """
    df such that theoretical excess kurtosis of t(df) = target_ek.
    ek = 6/(df-4)  →  df = 6/ek + 4
    """
    if target_ek <= 0:
        return _NCT_DF_CEIL
    df = 6.0 / target_ek + 4.0
    return float(np.clip(df, _NCT_DF_FLOOR, _NCT_DF_CEIL))


def _sample_nct_iterative(
    df:          float,
    target_skew: float,
    n:           int,
    rng:         np.random.Generator,
    max_iter:    int = 2,
) -> np.ndarray:
    """
    Sample nct(df, nc) with iterative nc correction so that
    the realised skew of the sample matches target_skew.
    """
    nc = float(np.clip(target_skew * np.sqrt(df / 2.0), -_NCT_NC_MAX, _NCT_NC_MAX))

    seed_seq = rng.integers(0, 2**31, size=max_iter + 1)

    for i in range(max_iter):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                z = stats.nct.rvs(df=df, nc=nc, loc=0, scale=1,
                                  size=n, random_state=int(seed_seq[i]))
                if not np.all(np.isfinite(z)):
                    raise ValueError
            except Exception:
                z = stats.t.rvs(df=df, loc=0, scale=1,
                                size=n, random_state=int(seed_seq[i]))
                return z

        actual_skew = float(stats.skew(z))
        skew_err    = target_skew - actual_skew
        if abs(skew_err) < 0.05:
            return z
        if abs(actual_skew) > 1e-4:
            nc = float(np.clip(nc * (target_skew / actual_skew), -_NCT_NC_MAX, _NCT_NC_MAX))
        else:
            nc = float(np.clip(nc + skew_err * np.sqrt(df / 2.0), -_NCT_NC_MAX, _NCT_NC_MAX))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            z = stats.nct.rvs(df=df, nc=nc, loc=0, scale=1,
                              size=n, random_state=int(seed_seq[-1]))
            if not np.all(np.isfinite(z)):
                raise ValueError
        except Exception:
            z = stats.t.rvs(df=df, loc=0, scale=1,
                            size=n, random_state=int(seed_seq[-1]))
    return z


def _skew_post_shift(z: np.ndarray, target_skew: float, clamp: float = 0.20) -> np.ndarray:
    """
    Two-iteration cubic skew-shift.

    Transform: w = z + a*(z^3 - 3*z)
    Leading-order effect: skew(w) ≈ skew(z) + 6*a  (standardised z)

    v54 Fix D: per-window clamp lowered from 0.30 → 0.20 (default) to
    reduce cubic distortion of the AR1 Hurst structure.
    The global post-topup pass in rolling_fit_generate uses clamp=0.30.

    After each shift, re-normalise to (mean=0, std=1).
    """
    if len(z) < 4:
        return z

    for _ in range(2):
        current_skew = float(stats.skew(z))
        skew_gap     = target_skew - current_skew
        if abs(skew_gap) < 0.02:
            break
        a = float(np.clip(skew_gap / 6.0, -clamp, clamp))
        w = z + a * (z ** 3 - 3.0 * z)
        w_std = float(np.std(w))
        if w_std > 1e-10:
            z = (w - float(np.mean(w))) / w_std
        else:
            z = w

    return z


def _kurtosis_topup(
    rets:       np.ndarray,
    target_ek:  float,
    jump_std:   float,
    skew_raw:   float,
    rng:        np.random.Generator,
    topup_frac: float = 0.50,   # v54: lowered from 0.60 → 0.50 (trigger earlier)
    max_jumps:  int   = 20,
) -> np.ndarray:
    """
    v54 Fix B: Additive jump injection to restore kurtosis.

    Key v54 change: ADDITIVE mode  out[ix] += jump_draws
    (v53 used replace mode: out[ix] = jump_draws which occasionally
    inverted skew when jstd was large and skew_bias ≈ 0)

    Additive: injected jumps amplify existing tail observations so
    std grows only marginally (already large in tail positions) while
    the 4th moment rises sharply.

    n_jumps formula: deficit * len / 20  (was / 30 in v53, more aggressive).
    """
    if len(rets) < 4:
        return rets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        realised_ek = float(stats.kurtosis(rets))

    if realised_ek >= topup_frac * target_ek:
        return rets

    ek_deficit = target_ek - realised_ek
    ret_std    = float(np.std(rets))
    # v54: denominator 20 (was 30) → inject more jumps per unit deficit
    n_jumps    = int(np.clip(np.ceil(ek_deficit * len(rets) / 20.0), 1, max_jumps))

    skew_bias  = float(np.clip(skew_raw * 0.3, -1.0, 1.0))
    eff_jstd   = float(max(jump_std, ret_std * 3.0))

    jump_draws = rng.normal(skew_bias * eff_jstd, eff_jstd, size=n_jumps)

    out      = rets.copy()
    # v54 Fix B: additive — add to existing tail positions (largest |ret|)
    large_ix = np.argsort(np.abs(out))[-n_jumps:]
    out[large_ix] += jump_draws

    return out


# ---------------------------------------------------------------------------
# v47: Volume helpers
# ---------------------------------------------------------------------------

def _fit_volume_params(df_ohlc: pd.DataFrame, log_rets: np.ndarray):
    if "Volume" not in df_ohlc.columns:
        return 0.0, 1.0, 0.0, 0.0

    raw_vol = df_ohlc["Volume"].values.astype(float)
    raw_vol = np.maximum(raw_vol, 1.0)
    log_v   = np.log(raw_vol)

    vol_log_mean = float(np.mean(log_v))
    vol_log_std  = float(max(np.std(log_v, ddof=1), 1e-6))

    n_ret = len(log_rets)
    if n_ret >= 4 and len(log_v) > n_ret:
        lv_aligned = log_v[-n_ret:]
        abs_ret    = np.abs(log_rets)
        X = np.column_stack([np.ones(n_ret), abs_ret])
        try:
            coefs, _, _, _ = np.linalg.lstsq(X, lv_aligned, rcond=None)
            beta = float(np.clip(coefs[1], 0.0, 20.0))
        except Exception:
            beta = 0.0
    else:
        beta = 0.0

    resid = log_v - vol_log_mean
    if len(resid) >= 4:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
        ar1 = float(np.clip(ar1 if np.isfinite(ar1) else 0.0, -0.9, 0.9))
    else:
        ar1 = 0.0

    return vol_log_mean, vol_log_std, beta, ar1


def _generate_volume(
    z_final:      np.ndarray,
    vol_log_mean: float,
    vol_log_std:  float,
    vol_ret_beta: float,
    vol_ar1:      float,
    rng:          np.random.Generator,
) -> np.ndarray:
    n = len(z_final)
    if n == 0:
        return np.array([], dtype=np.int64)

    u = rng.normal(0.0, vol_log_std, size=n)
    u = u + vol_ret_beta * np.abs(z_final)

    innov_sc = float(np.sqrt(max(1.0 - vol_ar1 ** 2, 0.0)))
    v = np.empty(n)
    v[0] = u[0]
    for i in range(1, n):
        v[i] = vol_ar1 * v[i - 1] + innov_sc * u[i]

    log_vol = vol_log_mean + v
    raw = np.exp(log_vol)
    raw = np.clip(raw, 1.0, 1e13)
    return np.round(raw).astype(np.int64)


# ---------------------------------------------------------------------------
# 1. FIT
# ---------------------------------------------------------------------------

_TARGET_EK_MAX = 30.0


def fit(
    df_history:       pd.DataFrame,
    apply_trend_bias: bool  = True,
    ek_global_floor:  float = 1.0,    # v53 Fix A: 3.0 → 1.0 (let true ek propagate)
    ek_oversample:    float = _EK_OVERSAMPLE,  # v53 Fix B: pre-amplify for pipeline decay
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
        ek       = float(stats.kurtosis(log_rets))
        skew_raw = float(stats.skew(log_rets))

    # v53 Fix A: floor lowered to 1.0 so true high-ek windows are not truncated
    # v53 Fix B: multiply by oversample factor to compensate pipeline kurtosis decay
    raw_ek    = float(np.clip(ek, ek_global_floor, _TARGET_EK_MAX))
    target_ek = float(np.clip(raw_ek * ek_oversample, ek_global_floor, _TARGET_EK_MAX))

    if apply_trend_bias:
        real_trend = float(np.sum(log_rets))
        trend_bias = float(np.sign(real_trend) * abs(ret_mu) * 0.3)
        ret_mu = ret_mu + trend_bias

    vol_log_mean, vol_log_std, vol_ret_beta, vol_ar1 = _fit_volume_params(df_history, log_rets)

    return StatParams(
        ret_mu          = ret_mu,
        ret_std         = ret_std,
        ret_skew_a      = skew_a,
        ret_skew_raw    = skew_raw,
        ret_df          = df_t,
        hurst_target    = float(np.clip(h, 0.3, 0.69)),
        wick_lambda     = wick_lam,
        atr_mean        = atr_mean,
        jump_freq       = jump_freq,
        jump_std        = jump_std,
        vol_persistence = vol_persistence,
        acf_lag1        = acf_lag1,
        target_ek       = target_ek,
        vol_log_mean    = vol_log_mean,
        vol_log_std     = vol_log_std,
        vol_ret_beta    = vol_ret_beta,
        vol_ar1         = vol_ar1,
    )


# ---------------------------------------------------------------------------
# 2. GENERATE  (v54)
#
# 管線：
#   Step 1  採樣 nct with iterative nc correction
#           df_nct derived from pre-amplified target_ek (v53 Fix B)
#   Step 2  AR1 Hurst 過濾
#   Step 3  ACF lag1 微調
#   Step 4  GARCH(1,1) clip (0.5, 2.0)
#   Step 5  Skew post-shift: 2-iter, clamp 0.20 (v54 Fix D: 0.30→0.20 per-window)
#   Step 6  jump 疊加
#   Step 7  exact rescale → (ret_mu, ret_std)
#   Step 8  OHLC wick
#   Step 9  Volume
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
    skew_raw        = float(params.get("ret_skew_raw", 0.0))
    hurst           = params["hurst_target"]
    wick_lam        = params["wick_lambda"]
    atr_mean        = params["atr_mean"]
    jump_freq       = params["jump_freq"]
    jump_std        = params["jump_std"]
    vol_persistence = params["vol_persistence"]
    acf_lag1        = params["acf_lag1"]
    target_ek       = params["target_ek"]

    vol_log_mean  = float(params.get("vol_log_mean", 0.0))
    vol_log_std   = float(params.get("vol_log_std",  1.0))
    vol_ret_beta  = float(params.get("vol_ret_beta", 0.0))
    vol_ar1       = float(params.get("vol_ar1",      0.0))

    total = n_bars + _AR1_WARMUP

    # ------------------------------------------------------------------
    # Step 1: Sample nct with iterative nc correction
    # ------------------------------------------------------------------
    df_nct = _df_from_ek(target_ek)
    z_raw  = _sample_nct_iterative(
        df=df_nct,
        target_skew=skew_raw,
        n=total,
        rng=rng,
        max_iter=2,
    )

    # ------------------------------------------------------------------
    # Step 2: AR1 Hurst
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
    # Step 3: ACF lag1 微調
    # ------------------------------------------------------------------
    if abs(acf_lag1) > 0.05 and n_bars > 1:
        z_acf = z_ar.copy()
        for i in range(1, n_bars):
            z_acf[i] = z_ar[i] + acf_lag1 * z_ar[i - 1] * 0.3
        z_s  = float(np.std(z_acf))
        z_ar = z_acf / z_s if z_s > 1e-10 else z_acf

    # ------------------------------------------------------------------
    # Step 4: GARCH(1,1) — clip (0.5, 2.0)
    # ------------------------------------------------------------------
    alpha = vol_persistence
    if alpha > 0.02 and n_bars > 1:
        beta_g   = float(np.clip(alpha * 0.8, 0.0, 0.89))
        alpha_s  = float(np.clip(alpha * 0.15, 0.01, 0.20))
        base_var = 1.0
        omega    = base_var * max(1.0 - beta_g - alpha_s, 0.01)

        h_t        = np.empty(n_bars)
        z_garch    = np.empty(n_bars)
        h_t[0]     = base_var
        z_garch[0] = z_ar[0]

        for i in range(1, n_bars):
            h_t[i]     = omega + beta_g * h_t[i - 1] + alpha_s * z_garch[i - 1] ** 2
            h_t[i]     = max(h_t[i], base_var * 0.01)
            vol_scale  = float(np.clip(np.sqrt(h_t[i] / base_var), 0.5, 2.0))
            z_garch[i] = z_ar[i] * vol_scale
    else:
        z_garch = z_ar

    # ------------------------------------------------------------------
    # Step 5: Skew post-shift (v54 Fix D) — 2-iter, clamp 0.20 per-window
    # ------------------------------------------------------------------
    z_shifted = _skew_post_shift(z_garch, target_skew=skew_raw, clamp=0.20)

    # ------------------------------------------------------------------
    # Step 6: jump 疊加 (ek-amplified + skew-biased)
    # ------------------------------------------------------------------
    z_work = z_shifted.copy()
    if jump_freq > 0 and n_bars > 0:
        n_jumps = int(rng.binomial(n_bars, jump_freq))
        if n_jumps > 0:
            jump_idx = rng.choice(n_bars, size=n_jumps, replace=False)
            ek_amp   = float(np.sqrt(max(target_ek / 6.0, 1.0)))
            eff_std  = jump_std * ek_amp
            skew_b   = float(np.clip(skew_raw * 0.15, -0.5, 0.5))
            z_work[jump_idx] += rng.normal(skew_b, eff_std, size=n_jumps)

    # ------------------------------------------------------------------
    # Step 7: exact rescale → (ret_mu, ret_std)
    # ------------------------------------------------------------------
    z_std = float(np.std(z_work))
    if z_std > 1e-10:
        z_final = (z_work - float(np.mean(z_work))) / z_std * ret_std + ret_mu
    else:
        z_final = np.full(n_bars, ret_mu)

    # ------------------------------------------------------------------
    # Step 8: OHLC
    # ------------------------------------------------------------------
    prices    = np.empty(n_bars + 1)
    prices[0] = start_price
    for i in range(n_bars):
        prices[i + 1] = prices[i] * np.exp(z_final[i])

    opens  = prices[:-1].copy()
    closes = prices[1:].copy()
    atr_adj = max(atr_mean, 1e-4)
    upper_w = rng.exponential(scale=wick_lam * atr_adj, size=n_bars)
    lower_w = rng.exponential(scale=wick_lam * atr_adj, size=n_bars)
    highs   = np.maximum(opens, closes) + upper_w
    lows    = np.maximum(np.minimum(opens, closes) - lower_w, 1e-6)

    # ------------------------------------------------------------------
    # Step 9: Volume
    # ------------------------------------------------------------------
    if vol_log_mean != 0.0 or vol_log_std != 1.0:
        volume = _generate_volume(
            z_final      = z_final,
            vol_log_mean = vol_log_mean,
            vol_log_std  = vol_log_std,
            vol_ret_beta = vol_ret_beta,
            vol_ar1      = vol_ar1,
            rng          = rng,
        )
    else:
        volume = np.zeros(n_bars, dtype=np.int64)

    return pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volume,
    })


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
        XtX  = X.T @ X + alpha * np.eye(d)
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
            0.0, self.max_blend,
        ))
        X     = np.array(self._X, dtype=float)
        x_new = np.array([
            params["ret_std"], params["ret_skew_a"],
            params["hurst_target"], params["target_ek"],
            params["ret_df"], params["vol_persistence"],
        ], dtype=float)
        new_std   = float(np.clip((1-blend)*params["ret_std"]      + blend*self._ridge_predict(X, np.array(self._y_std),   x_new), params["ret_std"]*0.3, params["ret_std"]*3.0))
        new_skew  = float(np.clip((1-blend)*params["ret_skew_a"]   + blend*self._ridge_predict(X, np.array(self._y_skew),  x_new), -10.0, 10.0))
        new_hurst = float(np.clip((1-blend)*params["hurst_target"] + blend*self._ridge_predict(X, np.array(self._y_hurst), x_new), 0.3, 0.69))
        new_ek    = float(np.clip((1-blend)*params["target_ek"]    + blend*self._ridge_predict(X, np.array(self._y_ek],    x_new), 1.0, _TARGET_EK_MAX))
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

    # ------------------------------------------------------------------
    # v54 Fix A: compute global skew anchor from full price series
    # Each window's skew_raw will be soft-clipped to this ± 0.5
    # so that aggregate skew after concatenation stays near the true value.
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _global_log_rets = _log_returns(df_real["Close"].values)
        global_skew_target = float(stats.skew(_global_log_rets)) if len(_global_log_rets) > 3 else 0.0

    _SKEW_ANCHOR_BAND = 0.5  # allow ±0.5 around global target per window

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

        # v54 Fix A: soft-clip ret_skew_raw to global anchor band
        raw_skew_window = float(params["ret_skew_raw"])
        anchored_skew   = float(np.clip(
            raw_skew_window,
            global_skew_target - _SKEW_ANCHOR_BAND,
            global_skew_target + _SKEW_ANCHOR_BAND,
        ))
        if anchored_skew != raw_skew_window:
            params = StatParams(**{**dict(params), "ret_skew_raw": anchored_skew})

        calib_action = None
        if calibrator is not None:
            from sim.calibrator import AdaptiveCalibrator
            ctx          = AdaptiveCalibrator.build_context(params)
            calib_action = calibrator.predict(ctx)
            params_dict  = calib_action.apply(dict(params))
            params       = StatParams(**params_dict)
        elif predictor is not None:
            params = predictor.predict_correction(params, window_idx)

        if result_chunks:
            start_price = float(result_chunks[-1]["Close"].iloc[-1])
        else:
            start_price = float(df_real["Open"].iloc[fwd_start])

        sim_seed = int(rng.integers(0, 2**31))
        df_sim   = generate(params=params, n_bars=fwd_bars,
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
            dtw_str   = f"{dtw_val:.4f}"    if np.isfinite(dtw_val)   else "  nan"
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

    # -----------------------------------------------------------------------
    # Global std-rescale (v48) + drift alignment (v49.2)
    # -----------------------------------------------------------------------
    df_result = pd.concat(result_chunks, ignore_index=True)

    n_real_tail = len(df_result)
    if n_real_tail > 1:
        real_tail_rets = np.diff(np.log(np.maximum(
            df_real["Close"].values[-n_real_tail:].astype(float), 1e-10
        )))
        sim_all_rets = np.diff(np.log(np.maximum(
            df_result["Close"].values.astype(float), 1e-10
        )))
        real_global_std = float(np.std(real_tail_rets))
        sim_global_std  = float(np.std(sim_all_rets))

        if sim_global_std > 1e-10 and real_global_std > 1e-10:
            scale = real_global_std / sim_global_std
            if 0.5 < scale < 2.0:
                orig_closes = df_result["Close"].values.astype(float)
                orig_opens  = df_result["Open"].values.astype(float)
                orig_highs  = df_result["High"].values.astype(float)
                orig_lows   = df_result["Low"].values.astype(float)

                log_ret_sim = np.diff(np.log(np.maximum(orig_closes, 1e-10)))
                scaled_rets = log_ret_sim * scale
                new_closes  = np.empty(len(orig_closes))
                new_closes[0] = orig_closes[0]
                for i in range(1, len(new_closes)):
                    new_closes[i] = new_closes[i - 1] * np.exp(scaled_rets[i - 1])

                price_ratio        = new_closes / np.maximum(orig_closes, 1e-10)
                df_result          = df_result.copy()
                df_result["Close"] = new_closes
                df_result["Open"]  = orig_opens * price_ratio
                df_result["High"]  = orig_highs * price_ratio
                df_result["Low"]   = orig_lows  * price_ratio

        sim_all_rets_post = np.diff(np.log(np.maximum(
            df_result["Close"].values.astype(float), 1e-10
        )))
        real_log_ret_mean = float(np.mean(real_tail_rets))
        sim_log_ret_mean  = float(np.mean(sim_all_rets_post))
        drift_correction  = real_log_ret_mean - sim_log_ret_mean

        if abs(drift_correction) < 0.005:
            final_closes    = df_result["Close"].values.astype(float)
            correction_path = np.exp(np.arange(len(final_closes)) * drift_correction)
            ratio           = correction_path / correction_path[0]
            df_result       = df_result.copy()
            df_result["Close"] = final_closes              * ratio
            df_result["Open"]  = df_result["Open"].values  * ratio
            df_result["High"]  = df_result["High"].values  * ratio
            df_result["Low"]   = df_result["Low"].values   * ratio

        # ------------------------------------------------------------------
        # v54: Kurtosis top-up (additive, topup_frac=0.50) after global rescale
        # followed by global skew correction pass (Fix B + Fix C)
        # ------------------------------------------------------------------
        final_rets = np.diff(np.log(np.maximum(
            df_result["Close"].values.astype(float), 1e-10
        )))
        mean_target_ek = float(np.mean([
            p.get("target_ek", 3.0)
            for p in param_log
            if not p.get("_summary") and p.get("target_ek") is not None
        ])) if param_log else 3.0

        mean_jump_std = float(np.mean([
            p.get("jump_std", float(np.std(final_rets) * 3.0))
            for p in param_log
            if not p.get("_summary") and p.get("jump_std") is not None
        ])) if param_log else float(np.std(final_rets) * 3.0)

        topup_rng = np.random.default_rng(seed + 9999 if seed is not None else 0)
        # Use raw (non-amplified) target ek for topup comparison
        mean_raw_target_ek = mean_target_ek / _EK_OVERSAMPLE
        final_rets_topup = _kurtosis_topup(
            rets       = final_rets,
            target_ek  = mean_raw_target_ek,
            jump_std   = mean_jump_std,
            skew_raw   = global_skew_target,   # v54: use global skew (not mean_skew_raw)
            rng        = topup_rng,
            topup_frac = 0.50,
            max_jumps  = 20,
        )

        # v54 Fix C: global skew correction pass after topup (clamp=0.30)
        final_rets_topup = _skew_post_shift(
            final_rets_topup,
            target_skew=global_skew_target,
            clamp=0.30,
        )

        if not np.allclose(final_rets_topup, final_rets, atol=1e-12):
            base_close  = df_result["Close"].values[0]
            new_closes  = np.empty(len(df_result))
            new_closes[0] = base_close
            for i in range(1, len(new_closes)):
                new_closes[i] = new_closes[i - 1] * np.exp(final_rets_topup[i - 1])
            price_ratio        = new_closes / np.maximum(df_result["Close"].values.astype(float), 1e-10)
            df_result          = df_result.copy()
            df_result["Close"] = new_closes
            df_result["Open"]  = df_result["Open"].values  * price_ratio
            df_result["High"]  = df_result["High"].values  * price_ratio
            df_result["Low"]   = df_result["Low"].values   * price_ratio

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
