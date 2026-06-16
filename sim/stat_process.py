"""
stat_process.py  v50
====================
純統計過程模型，完全不使用 agent。

v50: Revert Fleishman Power Transform (v49 family) — it destroyed skew/kurtosis
     and time-series structure (path_corr dropped to +0.11).

     Root cause: Fleishman is designed for i.i.d. samples; applying it after
     AR1/GARCH/jump reordered the innovations, wiping out autocorrelation.
     The exact-rescale after Fleishman also collapsed skew back toward zero.

     Replacement strategy for fat tails + skew:
       Step 1  Heavy-tail base: always use t(df) + skewnorm mixture.
               df is MLE-fitted per window; p_t driven by target_ek directly.
               NIG path kept for target_ek > 15 as a secondary option.
       Step 5b Jump amplification: when target_ek > 6, scale jump_std by
               sqrt(target_ek / 6) so sparse jumps carry fatter tails.
               Jump size drawn from skewed-t to also push skew.
     Everything else (AR1 Hurst, ACF, GARCH, exact rescale, drift alignment)
     is unchanged from v49.2.

v49.3: fix bracket mismatch in OnlineRidgePredictor.predict_correction
v49.2: fix rolling level drift — start_price chaining + global drift alignment
v49.1: fix Fleishman mean-drift & skew sign (3-unknown system)
v49: Step 6b Fleishman Power Transform (now removed)
v48.1: fix bracket mismatch
v48: std inflation, skew sign flip, kurtosis overshoot fixes
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


def _nig_params_from_moments(std, skew_raw, kurt_excess):
    if kurt_excess < 0.5:
        return None
    try:
        a_est = float(np.sqrt(3.0 / max(kurt_excess, 0.1)))
        a_est = float(np.clip(a_est, 0.15, 5.0))
        b_est = float(skew_raw * a_est / 3.0)
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
        skew_raw = float(stats.skew(log_rets))

    target_ek = float(max(np.clip(ek, 0.5, _TARGET_EK_MAX), ek_global_floor))

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
# 2. GENERATE  (v50: Fleishman removed; fat tails via t-mix + jump amplify)
#
# 管線：
#   Step 1  採樣基礎分佈（NIG if ek>15, else t+skewnorm mixture）
#           p_t = clip(target_ek / t_ek_implied, 0.10, 0.95) — drives fat tails
#   Step 2  AR1 Hurst 過濾
#   Step 3  ACF lag1 微調
#   Step 4  對稱 GARCH(1,1)
#   Step 5  jump 疊加
#   Step 5b Jump amplification: scale jump_std by sqrt(ek/6) when ek>6
#           Jump direction biased by skew_raw sign to push skew
#   Step 6  exact rescale → (ret_mu, ret_std)
#   Step 7  OHLC wick
#   Step 8  Volume
# ---------------------------------------------------------------------------

_AR1_WARMUP   = 50
_NIG_TRIGGER  = 15.0


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
    skew_raw        = float(params.get("ret_skew_raw", 0.0))
    df_t            = params["ret_df"]
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
    # Step 1: 採樣基礎分佈
    # Use low df_t (fat tails) and high p_t (t-fraction) to hit target_ek.
    # ------------------------------------------------------------------
    z_raw = None

    if target_ek > _NIG_TRIGGER:
        nig_ab = _nig_params_from_moments(ret_std, skew_raw, target_ek)
        if nig_ab is not None:
            a_nig, b_nig = nig_ab
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    z_raw = stats.norminvgauss.rvs(
                        a=a_nig, b=b_nig, loc=0, scale=1,
                        size=total, random_state=int(rng.integers(0, 2**31))
                    )
                    if not np.all(np.isfinite(z_raw)):
                        z_raw = None
                except Exception:
                    z_raw = None

    if z_raw is None:
        # df_t: MLE-fitted, but floor at 2.5 to ensure heavy tails exist.
        # p_t: fraction of t samples; higher ek → more t draws.
        # t_ek_implied: excess kurtosis of t(df) = 6/(df-4) for df>4.
        df_t_eff = float(np.clip(df_t, 2.5, 12.0))   # cap at 12 (not 20/30)
        if df_t_eff > 4.0:
            t_ek_implied = 6.0 / (df_t_eff - 4.0)
        else:
            t_ek_implied = target_ek  # df<=4: infinite kurtosis, use 100% t
        p_t = float(np.clip(target_ek / max(t_ek_implied, 0.5), 0.10, 0.95))
        mask  = rng.uniform(size=total) < p_t
        z_raw = np.where(
            mask,
            stats.t.rvs(df=df_t_eff, loc=0, scale=1, size=total,
                        random_state=int(rng.integers(0, 2**31))),
            stats.skewnorm.rvs(a=skew_a, loc=0, scale=1, size=total,
                               random_state=int(rng.integers(0, 2**31))),
        )

    # ------------------------------------------------------------------
    # Step 2: AR1 Hurst
    # ------------------------------------------------------------------
    rho = _ar1_hurst_rho(hurst)
    if abs(rho) > 1e-6:
        innov_sc = float(np.sqrt(max(1.0 - rho ** 2, 0.0)))
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
        z_s = float(np.std(z_acf))
        z_ar = z_acf / z_s if z_s > 1e-10 else z_acf

    # ------------------------------------------------------------------
    # Step 4: 對稱 GARCH(1,1)
    # ------------------------------------------------------------------
    alpha = vol_persistence
    if alpha > 0.02 and n_bars > 1:
        beta_g   = float(np.clip(alpha * 0.8, 0.0, 0.89))
        alpha_s  = float(np.clip(alpha * 0.15, 0.01, 0.20))
        base_var = 1.0
        omega    = base_var * max(1.0 - beta_g - alpha_s, 0.01)

        h_t      = np.empty(n_bars)
        z_garch  = np.empty(n_bars)
        h_t[0]   = base_var
        z_garch[0] = z_ar[0]

        for i in range(1, n_bars):
            h_t[i]     = omega + beta_g * h_t[i - 1] + alpha_s * z_garch[i - 1] ** 2
            h_t[i]     = max(h_t[i], base_var * 0.01)
            vol_scale  = float(np.clip(np.sqrt(h_t[i] / base_var), 0.7, 1.3))
            z_garch[i] = z_ar[i] * vol_scale
    else:
        z_garch = z_ar

    # ------------------------------------------------------------------
    # Step 5: jump 疊加
    # Step 5b: amplify jump_std when target_ek > 6 to reproduce fat tails;
    #          bias jump sign toward skew_raw direction.
    # ------------------------------------------------------------------
    z_work = z_garch.copy()
    if jump_freq > 0 and n_bars > 0:
        n_jumps = int(rng.binomial(n_bars, jump_freq))
        if n_jumps > 0:
            jump_idx = rng.choice(n_bars, size=n_jumps, replace=False)

            # Step 5b: ek-driven amplitude scaling
            ek_amp = float(np.sqrt(max(target_ek / 6.0, 1.0)))
            eff_jump_std = jump_std * ek_amp

            # skew bias: shift jump mean toward sign(skew_raw)
            skew_bias = float(np.clip(skew_raw * 0.15, -0.5, 0.5))
            jump_sizes = rng.normal(skew_bias, eff_jump_std, size=n_jumps)
            z_work[jump_idx] += jump_sizes

    # ------------------------------------------------------------------
    # Step 6: exact rescale → (ret_mu, ret_std)
    # ------------------------------------------------------------------
    z_std = float(np.std(z_work))
    if z_std > 1e-10:
        z_final = (z_work - float(np.mean(z_work))) / z_std * ret_std + ret_mu
    else:
        z_final = np.full(n_bars, ret_mu)

    # ------------------------------------------------------------------
    # Step 7: OHLC
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
    # Step 8: Volume
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

        # v49.2: chain start_price from previous sim chunk's last Close
        if result_chunks:
            start_price = float(result_chunks[-1]["Close"].iloc[-1])
        else:
            start_price = float(df_real["Open"].iloc[fwd_start])

        sim_seed = int(rng.integers(0, 2**31))
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

    # -----------------------------------------------------------------------
    # v48 fix: global std-rescale after concat
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
                new_closes = np.empty(len(orig_closes))
                new_closes[0] = orig_closes[0]
                for i in range(1, len(new_closes)):
                    new_closes[i] = new_closes[i - 1] * np.exp(scaled_rets[i - 1])

                price_ratio = new_closes / np.maximum(orig_closes, 1e-10)
                df_result = df_result.copy()
                df_result["Close"] = new_closes
                df_result["Open"]  = orig_opens  * price_ratio
                df_result["High"]  = orig_highs  * price_ratio
                df_result["Low"]   = orig_lows   * price_ratio

        # -------------------------------------------------------------------
        # v49.2: global drift alignment
        # -------------------------------------------------------------------
        sim_all_rets_post = np.diff(np.log(np.maximum(
            df_result["Close"].values.astype(float), 1e-10
        )))
        real_log_ret_mean = float(np.mean(real_tail_rets))
        sim_log_ret_mean  = float(np.mean(sim_all_rets_post))
        drift_correction  = real_log_ret_mean - sim_log_ret_mean

        if abs(drift_correction) < 0.005:
            final_closes = df_result["Close"].values.astype(float)
            correction_path = np.exp(np.arange(len(final_closes)) * drift_correction)
            ratio = correction_path / correction_path[0]
            df_result = df_result.copy()
            df_result["Close"] = final_closes * ratio
            df_result["Open"]  = df_result["Open"].values  * ratio
            df_result["High"]  = df_result["High"].values  * ratio
            df_result["Low"]   = df_result["Low"].values   * ratio

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
