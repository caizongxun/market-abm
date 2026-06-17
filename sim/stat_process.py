"""
stat_process.py  v72
====================
v72: fix global-pass kurtosis destruction

  Root cause analysis:
  The global post-processing pass in rolling_fit_generate() was destroying
  kurtosis in three sequential steps:
    1. std rescale (real/sim) — changes vol distribution uniformly, OK
    2. _soft_clip_z — compresses tails of z-scores, directly kills kurtosis
    3. _linear_skew_align(z_wins) — operates on already-clipped z, further smooths
    4. final rescale by real_global_std — renormalises but kurtosis is already gone

  The real culprit is step 2: sc_inner=3.0 means ANY return > 3*sigma is
  soft-clipped. For AAPL with true kurtosis ~10.6, the "fat tail" events
  ARE the kurtosis — clipping them at 3-sigma is equivalent to Winsorising
  the entire kurtosis signal. sc_inner needs to be set relative to target_ek,
  not as a fixed sigma threshold.

  Fixes:
  A. _soft_clip_z: set inner = max(5.0, 2*sqrt(target_ek)) so for ek=10.6
     inner ≈ 6.5 sigma — tails are only clipped at extreme outliers, not fat
     tail events. outer = inner + 3.0 (was 2x, now additive to avoid too
     wide a range that defeats the purpose).
  B. After _soft_clip_z, inject kurtosis back with _kurtosis_topup() using
     the global true_target_ek (median of window target_ek values).
  C. _linear_skew_align: apply BEFORE the final std pin, not on z_wins
     (which has lost its tails). Operate on final_rets_fixed after topup.
  D. Remove the global std rescale step that pins to real_global_std AFTER
     _linear_skew_align — this was the final kurtosis-destroying step because
     it re-standardises a distribution that has just been topup'd.
     Instead, pin std precisely without touching the tail structure.

v71: fix generate_conditional degradation
v70: kurtosis p90 fix + block-bootstrap conditional generate
v69: Fix module-global _ek_decay_ema cross-trial state leak.
v68: Pass fwd_bars to calibrator.record() for kurt sample-size discount.
v67: Per-symbol dynamic soft_clip inner/outer based on median target_ek.
v66 (P1b): Per-window realised-kurtosis feedback into calibrator context.
v65 (P1): _TARGET_EK_MAX 30->60, ek_oversample_adj clip upper 6->10.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

try:
    from sim.calibrator import AdaptiveCalibrator  # noqa: F401
except ImportError:
    pass

# ---------------------------------------------------------------------------
# StatParams
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatParams:
    ret_mu:           float
    ret_std:          float
    ret_skew_a:       float
    ret_skew_raw:     float
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

    def __getitem__(self, key: str):
        return getattr(self, key)

    def keys(self):
        return self.__dataclass_fields__.keys()

    def __iter__(self):
        return iter(self.__dataclass_fields__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _log_returns(closes: np.ndarray) -> np.ndarray:
    c = np.maximum(closes.astype(float), 1e-10)
    return np.diff(np.log(c))


def _fit_df_scan(log_rets: np.ndarray) -> float:
    std = float(np.std(log_rets)) + 1e-10
    z   = log_rets / std
    best_df, best_ll = 30.0, -np.inf
    for df in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0]:
        ll = float(np.sum(stats.t.logpdf(z, df=df)))
        if ll > best_ll:
            best_ll, best_df = ll, df
    return best_df


def _fit_skewnorm(log_rets: np.ndarray) -> Tuple[float, float, float]:
    try:
        a, loc, scale = stats.skewnorm.fit(log_rets, floc=0)
        a = float(np.clip(a, -5.0, 5.0))
        return a, float(loc), float(scale)
    except Exception:
        return 0.0, 0.0, float(np.std(log_rets))


def hurst_exponent(ts: np.ndarray, min_len: int = 20) -> float:
    ts = np.asarray(ts, dtype=float)
    n  = len(ts)
    if n < min_len:
        return 0.5
    lags, rs_vals = [], []
    for lag in [max(4, n // 8), max(8, n // 4), max(16, n // 2)]:
        if lag >= n:
            continue
        chunks = [ts[i:i+lag] for i in range(0, n - lag + 1, lag)]
        if not chunks:
            continue
        rs_chunk = []
        for c in chunks:
            c = c - c.mean()
            cs = np.cumsum(c)
            r  = cs.max() - cs.min()
            s  = float(np.std(c)) + 1e-10
            rs_chunk.append(r / s)
        if rs_chunk:
            lags.append(np.log(lag))
            rs_vals.append(np.log(np.mean(rs_chunk)))
    if len(lags) < 2:
        return 0.5
    h = float(np.polyfit(lags, rs_vals, 1)[0])
    return float(np.clip(h, 0.1, 0.9))


def _fit_wick_lambda(df: pd.DataFrame) -> Tuple[float, float]:
    hi = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    cl = df["Close"].values.astype(float)
    op = df["Open"].values.astype(float)
    body   = np.abs(cl - op)
    range_ = hi - lo + 1e-10
    wick_r = body / range_
    wick_r = np.clip(wick_r, 1e-3, 1.0)
    lam = float(np.clip(-np.mean(np.log(wick_r)), 0.1, 5.0))
    atr = float(np.mean(range_ / (cl + 1e-10)))
    return lam, atr


def _fit_jump_params(log_rets: np.ndarray, ret_std: float) -> Tuple[float, float]:
    threshold  = 2.5 * ret_std
    jump_mask  = np.abs(log_rets) > threshold
    jump_freq  = float(np.mean(jump_mask))
    jump_vals  = log_rets[jump_mask]
    jump_std   = float(np.std(jump_vals)) if len(jump_vals) > 1 else ret_std * 3.0
    return float(np.clip(jump_freq, 0.0, 0.20)), float(np.clip(jump_std, ret_std, ret_std * 10.0))


def _fit_vol_persistence(log_rets: np.ndarray, ret_mu: float) -> float:
    sq = (log_rets - ret_mu) ** 2
    if len(sq) < 3:
        return 0.0
    try:
        corr = float(np.corrcoef(sq[:-1], sq[1:])[0, 1])
        return float(np.clip(corr, 0.0, 0.85))
    except Exception:
        return 0.0


def _fit_acf_lag1(log_rets: np.ndarray) -> float:
    if len(log_rets) < 3:
        return 0.0
    try:
        return float(np.clip(np.corrcoef(log_rets[:-1], log_rets[1:])[0, 1], -0.5, 0.5))
    except Exception:
        return 0.0


def _fit_volume_params(df: pd.DataFrame, log_rets: np.ndarray):
    if "Volume" not in df.columns:
        return 0.0, 0.5, 0.0, 0.5
    vol = df["Volume"].values.astype(float)
    vol = np.maximum(vol, 1.0)
    log_vol = np.log(vol)
    log_vol_mean = float(np.mean(log_vol))
    log_vol_std  = float(np.std(log_vol)) + 1e-6
    min_len = min(len(log_rets), len(log_vol) - 1)
    if min_len > 1:
        vol_ret_beta = float(np.clip(
            np.corrcoef(np.abs(log_rets[:min_len]), log_vol[1:min_len+1])[0, 1],
            -1.0, 1.0,
        ))
    else:
        vol_ret_beta = 0.0
    if len(log_vol) > 2:
        vol_ar1 = float(np.clip(np.corrcoef(log_vol[:-1], log_vol[1:])[0, 1], -1.0, 1.0))
    else:
        vol_ar1 = 0.5
    return log_vol_mean, log_vol_std, vol_ret_beta, vol_ar1


# ---------------------------------------------------------------------------
# 1. FIT
# ---------------------------------------------------------------------------

_TARGET_EK_MAX    = 60.0
_EK_OVERSAMPLE    = 3.2
_HIGH_KURT_THRESHOLD = 8.0


def fit(
    df_history:       pd.DataFrame,
    apply_trend_bias: bool  = True,
    ek_global_floor:  float = 1.0,
    ek_oversample:    float = _EK_OVERSAMPLE,
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
    raw_ek = float(np.clip(ek, ek_global_floor, _TARGET_EK_MAX))

    if raw_ek > _HIGH_KURT_THRESHOLD:
        boost = 1.0 + 0.5 * float(np.clip((raw_ek - _HIGH_KURT_THRESHOLD) / (2 * _HIGH_KURT_THRESHOLD), 0.0, 1.0))
        ek_oversample = ek_oversample * boost

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
        hurst_target    = float(np.clip(h, 0.30, 0.69)),
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
# 2. GENERATE  (parametric)
# ---------------------------------------------------------------------------

def _df_from_ek(target_ek: float) -> float:
    if target_ek <= 0:
        return 30.0
    df = 6.0 / target_ek + 4.0
    return float(np.clip(df, 2.1, 30.0))


def _simulate_garch_vol(
    n:              int,
    base_vol:       float,
    persistence:    float,
    acf_lag1:       float,
    rng:            np.random.Generator,
) -> np.ndarray:
    omega  = base_vol ** 2 * (1.0 - persistence)
    alpha  = persistence * 0.3
    beta   = persistence * 0.7
    vols   = np.empty(n)
    h_prev = base_vol ** 2
    for i in range(n):
        h_prev   = max(omega + alpha * h_prev + beta * h_prev, 1e-12)
        vols[i]  = np.sqrt(h_prev)
    return vols


def _kurtosis_topup(
    rets:       np.ndarray,
    target_ek:  float,
    rng:        np.random.Generator,
    topup_frac: float = 0.60,
    max_iter:   int   = 8,
) -> np.ndarray:
    """
    Inject tail events until realised_ek >= topup_frac * target_ek.
    Uses a copy so caller's array is not mutated unexpectedly.
    """
    rets = rets.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        realised_ek = float(stats.kurtosis(rets))
    if realised_ek >= topup_frac * target_ek:
        return rets
    n_total = len(rets)
    std_r   = float(np.std(rets)) + 1e-10
    # use ~1% of bars per iteration, at least 1
    n_jumps = max(1, int(n_total * 0.01))
    for _ in range(max_iter):
        ek_deficit = max(target_ek - realised_ek, 1.0)
        idx   = rng.choice(n_total, size=n_jumps, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_jumps)
        rets[idx] += signs * std_r * float(np.sqrt(ek_deficit))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            realised_ek = float(stats.kurtosis(rets))
        if realised_ek >= topup_frac * target_ek:
            break
    return rets


def _soft_clip_z(
    z:      np.ndarray,
    inner:  float = 5.0,
    outer:  float = 8.0,
) -> np.ndarray:
    """
    v72: inner/outer are now passed in per-call based on target_ek.
    Defaults raised so fat-tail events are not destroyed by default.
    Only returns beyond `outer` sigma are hard-clipped.
    """
    abs_z = np.abs(z)
    mask  = abs_z > inner
    if not np.any(mask):
        return z
    result = z.copy()
    tail   = abs_z[mask]
    compressed = inner + (outer - inner) * np.tanh((tail - inner) / (outer - inner))
    result[mask] = np.sign(z[mask]) * compressed
    return result


def _linear_skew_align(z: np.ndarray, target_skew: float, tol: float = 0.05) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        current_skew = float(stats.skew(z))
    if abs(current_skew - target_skew) < tol:
        return z
    shift = float(np.clip((target_skew - current_skew) * 0.5, -0.5, 0.5))
    return z + shift * np.abs(z)


def generate(
    params:     StatParams,
    n_bars:     int,
    start_price: float = 100.0,
    rng:         Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng()

    n         = n_bars
    ret_mu    = float(params["ret_mu"])
    ret_std   = float(params["ret_std"])
    skew_a    = float(params["ret_skew_a"])
    df_t      = float(params["ret_df"])
    h         = float(params["hurst_target"])
    wick_lam  = float(params["wick_lambda"])
    atr_mean  = float(params["atr_mean"])
    jump_freq = float(params["jump_freq"])
    jump_std  = float(params["jump_std"])
    vp        = float(params["vol_persistence"])
    acf_lag1  = float(params["acf_lag1"])
    target_ek = float(params["target_ek"])

    df_nct = _df_from_ek(target_ek)
    t_rets = stats.t.rvs(df=df_nct, size=n, random_state=rng) * ret_std + ret_mu

    if abs(skew_a) > 0.05:
        skn = stats.skewnorm.rvs(a=skew_a, size=n, random_state=rng)
        skn = (skn - np.mean(skn)) / (np.std(skn) + 1e-10) * ret_std
        blend = float(np.clip(abs(skew_a) / 5.0, 0.0, 0.5))
        t_rets = (1.0 - blend) * t_rets + blend * skn

    vols = _simulate_garch_vol(n, ret_std, vp, acf_lag1, rng)
    vol_scale = vols / (ret_std + 1e-10)
    t_rets = t_rets * vol_scale

    if jump_freq > 0 and jump_std > 0:
        jump_mask = rng.random(n) < jump_freq
        n_jumps   = int(jump_mask.sum())
        if n_jumps > 0:
            jumps = stats.t.rvs(df=3.0, size=n_jumps, random_state=rng) * jump_std
            t_rets[jump_mask] += jumps

    t_rets = _kurtosis_topup(t_rets, target_ek, rng)

    if abs(acf_lag1) > 0.02 and n > 5:
        ar_rets = np.empty(n)
        ar_rets[0] = t_rets[0]
        for i in range(1, n):
            ar_rets[i] = acf_lag1 * ar_rets[i - 1] + np.sqrt(max(1.0 - acf_lag1**2, 0.01)) * t_rets[i]
        t_rets = ar_rets

    actual_std = float(np.std(t_rets)) + 1e-10
    t_rets = t_rets / actual_std * ret_std

    prices = np.empty(n + 1)
    prices[0] = start_price
    for i in range(n):
        prices[i + 1] = prices[i] * np.exp(t_rets[i])
    prices = np.maximum(prices, 1e-4)

    opens  = prices[:-1]
    closes = prices[1:]

    raw_range = np.abs(closes - opens) / (np.exp(stats.expon.rvs(scale=1.0/wick_lam, size=n, random_state=rng)) + 1e-6)
    raw_range = np.clip(raw_range, 0.0, opens * 0.20)
    half = raw_range / 2.0
    highs = np.maximum(opens, closes) + half
    lows  = np.minimum(opens, closes) - half
    lows  = np.maximum(lows, opens * 0.01)

    vol_log_mean  = float(params["vol_log_mean"])
    vol_log_std   = float(params["vol_log_std"])
    vol_ret_beta  = float(params["vol_ret_beta"])
    vol_ar1       = float(params["vol_ar1"])
    base_log_vol  = rng.normal(0, 1, n)
    ar_log_vol    = np.empty(n)
    ar_log_vol[0] = base_log_vol[0]
    for i in range(1, n):
        ar_log_vol[i] = vol_ar1 * ar_log_vol[i-1] + np.sqrt(max(1 - vol_ar1**2, 0.01)) * base_log_vol[i]
    abs_rets = np.abs(t_rets)
    abs_rets_norm = (abs_rets - abs_rets.mean()) / (abs_rets.std() + 1e-10)
    log_vol_sim = vol_log_mean + vol_log_std * (ar_log_vol + vol_ret_beta * abs_rets_norm)
    volumes = np.maximum(np.exp(log_vol_sim), 1.0).astype(int)

    return pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    })


# ---------------------------------------------------------------------------
# 2b. GENERATE_CONDITIONAL
# ---------------------------------------------------------------------------

def _detect_regime(log_rets: np.ndarray, n_recent: int = 10) -> Dict[str, Any]:
    if len(log_rets) < n_recent + 5:
        return {"hurst": 0.5, "vol_ratio": 1.0, "momentum": 0, "vol_level": float(np.std(log_rets))}
    recent = log_rets[-n_recent:]
    global_vol = float(np.std(log_rets)) + 1e-10
    recent_vol = float(np.std(recent)) + 1e-10
    h = hurst_exponent(log_rets, min_len=20)
    return {
        "hurst":      h,
        "vol_ratio":  recent_vol / global_vol,
        "momentum":   int(np.sign(float(np.sum(recent)))),
        "vol_level":  recent_vol,
    }


def _score_block_regime_match(
    block_rets: np.ndarray,
    target_regime: Dict[str, Any],
) -> float:
    if len(block_rets) < 3:
        return 1e9
    h_block    = hurst_exponent(block_rets, min_len=4) if len(block_rets) >= 8 else 0.5
    vol_block  = float(np.std(block_rets)) + 1e-10
    mom_block  = int(np.sign(float(np.sum(block_rets))))
    vol_target = float(target_regime["vol_level"]) + 1e-10

    d_hurst = abs(h_block - float(target_regime["hurst"])) / 0.4
    d_vol   = abs(vol_block / vol_target - float(target_regime["vol_ratio"])) / 1.0
    d_mom   = 0.0 if mom_block == int(target_regime["momentum"]) or target_regime["momentum"] == 0 else 0.5
    return d_hurst + 0.5 * d_vol + 0.3 * d_mom


def generate_conditional(
    df_history:  pd.DataFrame,
    params:      StatParams,
    n_bars:      int,
    start_price: float = 100.0,
    rng:         Optional[np.random.Generator] = None,
    block_size:  int   = 10,
    n_candidates: int  = 30,
    bootstrap_weight: float = 0.50,
    regime_window: int = 15,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng()

    hist_closes = df_history["Close"].values.astype(float)
    hist_rets   = _log_returns(hist_closes)

    regime = _detect_regime(hist_rets, n_recent=regime_window)

    n_hist = len(hist_rets)
    if n_hist < block_size + 2:
        return generate(params, n_bars, start_price=start_price, rng=rng)

    scores: List[float] = []
    block_starts: List[int] = []
    for i in range(0, n_hist - block_size, max(1, block_size // 2)):
        b = hist_rets[i:i + block_size]
        s = _score_block_regime_match(b, regime)
        scores.append(s)
        block_starts.append(i)

    if not scores:
        return generate(params, n_bars, start_price=start_price, rng=rng)

    scores_arr = np.array(scores, dtype=float)
    inv_scores = np.exp(-scores_arr * 3.0)
    inv_scores = inv_scores / (inv_scores.sum() + 1e-10)

    n_blocks_needed = n_bars // block_size + 2
    chosen_starts = rng.choice(
        len(block_starts),
        size=n_blocks_needed,
        replace=True,
        p=inv_scores,
    )
    bootstrap_rets = np.concatenate(
        [hist_rets[block_starts[k]:block_starts[k] + block_size] for k in chosen_starts]
    )[:n_bars]

    target_std  = float(params["ret_std"])
    target_mu   = float(params["ret_mu"])
    target_ek   = float(params["target_ek"])
    target_skew = float(params["ret_skew_raw"])

    bs_std = float(np.std(bootstrap_rets)) + 1e-10
    bootstrap_rets = bootstrap_rets / bs_std * target_std
    bootstrap_rets = bootstrap_rets - float(np.mean(bootstrap_rets)) + target_mu

    df_param = generate(params, n_bars, start_price=start_price, rng=rng)
    param_closes = df_param["Close"].values.astype(float)
    param_rets   = _log_returns(param_closes)
    if len(param_rets) < n_bars:
        param_rets = np.pad(param_rets, (0, n_bars - len(param_rets)), mode="edge")

    w_b = bootstrap_weight
    w_p = 1.0 - bootstrap_weight
    blended_rets = w_b * bootstrap_rets + w_p * param_rets[:n_bars]

    bl_std = float(np.std(blended_rets)) + 1e-10
    blended_rets = blended_rets / bl_std * target_std
    blended_rets = blended_rets - float(np.mean(blended_rets)) + target_mu

    # recover kurtosis lost by linear blend
    blended_rets = _kurtosis_topup(blended_rets, target_ek, rng)

    # correct skew bias from block selection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        current_skew = float(stats.skew(blended_rets))
    if abs(current_skew - target_skew) > 0.15:
        blended_rets = _linear_skew_align(blended_rets, target_skew)

    # final std/mean pin
    bl_std2 = float(np.std(blended_rets)) + 1e-10
    blended_rets = blended_rets / bl_std2 * target_std
    blended_rets = blended_rets - float(np.mean(blended_rets)) + target_mu

    prices = np.empty(n_bars + 1)
    prices[0] = start_price
    for i in range(n_bars):
        prices[i + 1] = prices[i] * np.exp(blended_rets[i])
    prices = np.maximum(prices, 1e-4)

    opens  = prices[:-1]
    closes = prices[1:]

    price_ratio = closes / np.maximum(param_closes[:n_bars], 1e-10)
    highs   = df_param["High"].values[:n_bars].astype(float)   * price_ratio
    lows    = df_param["Low"].values[:n_bars].astype(float)    * price_ratio
    lows    = np.maximum(lows, opens * 0.01)
    volumes = df_param["Volume"].values[:n_bars]

    return pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    })


# ---------------------------------------------------------------------------
# 3. ONLINE RIDGE PREDICTOR
# ---------------------------------------------------------------------------

class OnlineRidgePredictor:
    def __init__(self, alpha: float = 1.0):
        self._alpha    = alpha
        self._X:       List[List[float]] = []
        self._y_std:   List[float]       = []
        self._y_skew:  List[float]       = []
        self._y_hurst: List[float]       = []
        self._y_ek:    List[float]       = []
        self._n_train: int = 0

    def _build_features(self, params: StatParams, window_idx: int) -> List[float]:
        return [
            float(params["ret_std"]),
            float(params["hurst_target"]),
            float(np.log1p(abs(params["target_ek"]))),
            float(params["vol_persistence"]),
            float(params["acf_lag1"]),
            float(np.sin(window_idx * 0.1)),
        ]

    def record(self, params, realised_std, realised_skew, realised_hurst, realised_ek,
               ek_oversample: float = _EK_OVERSAMPLE):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = [
                float(params["ret_std"]),
                float(params["hurst_target"]),
                float(np.log1p(abs(params["target_ek"]))),
                float(params["vol_persistence"]),
                float(params["acf_lag1"]),
                float(len(self._X)),
            ]
        self._X.append(x)
        self._y_std.append(float(realised_std))
        self._y_skew.append(float(realised_skew))
        self._y_hurst.append(float(realised_hurst))
        self._y_ek.append(realised_ek * ek_oversample)
        self._n_train += 1

    def _ridge_predict(self, X: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> float:
        if len(y) < 3:
            return float(np.mean(y)) if len(y) > 0 else 0.0
        Xm = np.atleast_2d(X)
        n, d = Xm.shape
        A = Xm.T @ Xm + self._alpha * np.eye(d)
        try:
            w = np.linalg.solve(A, Xm.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, Xm.T @ y, rcond=None)[0]
        return float(np.dot(x_new, w))

    def predict_correction(self, params, window_idx):
        if self._n_train < 3:
            return params
        X     = np.array(self._X, dtype=float)
        x_new = np.array(self._build_features(params, window_idx), dtype=float)
        n     = len(X)
        blend = float(np.clip((n - 3) / 10.0, 0.0, 0.40))

        new_std   = float(np.clip((1-blend)*params["ret_std"]      + blend*self._ridge_predict(X, np.array(self._y_std),   x_new), 1e-5, 0.20))
        new_hurst = float(np.clip((1-blend)*params["hurst_target"] + blend*self._ridge_predict(X, np.array(self._y_hurst), x_new), 0.30, 0.69))
        new_ek    = float(np.clip((1-blend)*params["target_ek"]    + blend*self._ridge_predict(X, np.array(self._y_ek),    x_new), 1.0, _TARGET_EK_MAX))

        return StatParams(**{**dict(params),
                             "ret_std": new_std,
                             "hurst_target": new_hurst, "target_ek": new_ek})


# ---------------------------------------------------------------------------
# 4. ROLLING FIT + GENERATE
# ---------------------------------------------------------------------------

_EK_DECAY_EMA_ALPHA = 0.20
_SOFT_CLIP_INNER    = 3.0   # kept for ek_decay feedback only, not for final pass
_SOFT_CLIP_OUTER    = 6.0
_SKEW_ANCHOR_BAND   = 0.30


def rolling_fit_generate(
    df_real:    pd.DataFrame,
    lookback:   int   = 60,
    step:       int   = 20,
    seed:       int   = 42,
    verbose:    bool  = True,
    calibrator          = None,
    predictor           = None,
    ek_decay_ema_init:  float = 1.0,
    use_conditional:    bool  = False,
    bootstrap_weight:   float = 0.50,
) -> Tuple[pd.DataFrame, List[Dict]]:
    ek_decay_ema: float = float(ek_decay_ema_init)

    rng = np.random.default_rng(seed)

    n = len(df_real)
    if n < lookback + step:
        raise ValueError(f"df_real too short ({n} rows), need >= {lookback + step}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _global_log_rets = _log_returns(df_real["Close"].values)
        global_skew_target = float(stats.skew(_global_log_rets)) if len(_global_log_rets) > 3 else 0.0

    result_chunks: List[pd.DataFrame] = []
    param_log:     List[Dict]         = []

    all_dtw:    List[float] = []
    all_pcorr:  List[float] = []

    window_idx = 0
    pos        = 0

    while pos + lookback + step <= n:
        fit_start = pos
        fit_end   = pos + lookback
        fwd_start = fit_end
        fwd_end   = min(fwd_start + step, n)
        fwd_bars  = fwd_end - fwd_start

        df_fit = df_real.iloc[fit_start:fit_end].copy().reset_index(drop=True)
        df_fwd = df_real.iloc[fwd_start:fwd_end].copy().reset_index(drop=True)

        ek_oversample_adj = float(np.clip(
            _EK_OVERSAMPLE / max(ek_decay_ema, 0.15),
            1.5, 12.0,
        ))

        params = fit(df_fit, ek_oversample=ek_oversample_adj)

        raw_skew_window = float(params["ret_skew_raw"])
        anchored_skew   = float(np.clip(
            raw_skew_window,
            global_skew_target - _SKEW_ANCHOR_BAND,
            global_skew_target + _SKEW_ANCHOR_BAND,
        ))
        if anchored_skew != raw_skew_window:
            params = StatParams(**{**dict(params), "ret_skew_raw": anchored_skew})

        calib_action = None
        ek_oversample_init = ek_oversample_adj
        if calibrator is not None:
            from sim.calibrator import AdaptiveCalibrator

            _fit_log_rets = _log_returns(df_fit["Close"].values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _realised_ek = float(stats.kurtosis(_fit_log_rets))
            _ek_needed = max(_realised_ek, 1.0)
            _ek_produced = max(float(params["target_ek"]) / max(ek_oversample_adj, 1.0), 1.0)
            ek_oversample_init = float(np.clip(
                max(ek_oversample_adj, _ek_needed / _ek_produced),
                1.0, 12.0,
            ))

            ctx          = AdaptiveCalibrator.build_context(params, ek_oversample=ek_oversample_init)
            calib_action = calibrator.predict(ctx)
            params_dict  = calib_action.apply(dict(params))
            params       = StatParams(**params_dict)
        elif predictor is not None:
            params = predictor.predict_correction(params, window_idx)

        if result_chunks:
            start_price = float(result_chunks[-1]["Close"].iloc[-1])
        else:
            start_price = float(df_real["Open"].iloc[fwd_start])

        if use_conditional:
            df_sim = generate_conditional(
                df_history       = df_real.iloc[:fit_end].copy().reset_index(drop=True),
                params           = params,
                n_bars           = fwd_bars,
                start_price      = start_price,
                rng              = rng,
                bootstrap_weight = bootstrap_weight,
            )
        else:
            df_sim = generate(params, n_bars=fwd_bars, start_price=start_price, rng=rng)

        result_chunks.append(df_sim)

        sim_rets  = np.diff(np.log(np.maximum(df_sim["Close"].values,  1e-10)))
        real_rets = np.diff(np.log(np.maximum(df_fwd["Close"].values,  1e-10)))

        real_std_w  = float(np.std(real_rets)) + 1e-10
        std_err_pct = abs(float(np.std(sim_rets)) / real_std_w - 1.0) if len(sim_rets) > 1 else float("nan")
        kurt_err    = abs(float(stats.kurtosis(sim_rets)) - float(stats.kurtosis(real_rets))) if len(sim_rets) > 3 else float("nan")
        hurst_err   = abs(float(hurst_exponent(sim_rets)) - float(hurst_exponent(real_rets))) if len(sim_rets) > 10 else float("nan")
        min_len     = min(len(real_rets), len(sim_rets))
        dir_hit     = float(np.mean(np.sign(real_rets[:min_len]) == np.sign(sim_rets[:min_len]))) if min_len > 0 else float("nan")

        if len(sim_rets) > 1 and len(real_rets) > 1:
            dtw_val = (
                abs(np.std(sim_rets) - np.std(real_rets)) / real_std_w
                + abs(np.mean(sim_rets) - np.mean(real_rets)) / real_std_w
            )
        else:
            dtw_val = float("nan")

        pcorr_val = float(np.corrcoef(
            real_rets / (np.std(real_rets) + 1e-10),
            sim_rets  / (np.std(sim_rets)  + 1e-10),
        )[0, 1]) if len(sim_rets) > 2 and len(real_rets) > 2 else float("nan")

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
                    n_bars      = fwd_bars,
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
                    ek_oversample=ek_oversample_adj,
                )

        real_c = float(df_fwd["Close"].iloc[-1])
        sim_c  = float(df_sim["Close"].iloc[-1])
        c_err  = (sim_c - real_c) / (real_c + 1e-10)

        if verbose:
            kurt_str  = f"{kurt_err:.2f}" if np.isfinite(kurt_err) else "N/A"
            calib_str = f"  [calib n={calibrator.n_experiences}]" if calibrator else ""
            cond_str  = "  [cond]" if use_conditional else ""
            print(
                f"[stat] w{window_idx:3d}"
                f"  std={params['ret_std']:.4f}  ek={params['target_ek']:.2f}"
                f"  ek_adj={ek_oversample_adj:.2f}  ek_init={ek_oversample_init if calibrator else '---'}"
                f"  kurt_err={kurt_str}"
                f"  c_err={c_err:+.3f}"
                f"{calib_str}{cond_str}"
            )

        param_log.append({
            "window":            window_idx,
            "fit_start":         fit_start,
            "fit_end":           fit_end,
            "ret_std":           float(params["ret_std"]),
            "ret_skew_a":        float(params["ret_skew_a"]),
            "hurst":             float(params["hurst_target"]),
            "target_ek":         float(params["target_ek"]),
            "ek_oversample_adj": ek_oversample_adj,
            "ek_oversample_init": ek_oversample_init if calibrator else None,
            "jump_freq":         float(params["jump_freq"]),
            "vol_persistence":   float(params["vol_persistence"]),
            "std_err_pct":       std_err_pct,
            "kurt_err":          kurt_err,
            "hurst_err":         hurst_err if np.isfinite(hurst_err) else None,
            "dir_hit":           dir_hit,
            "c_err":             c_err,
            "use_conditional":   use_conditional,
        })

        window_idx += 1
        pos        += step

    # ---------------------------------------------------------------------------
    # Global post-processing pass  (v72: kurtosis-preserving)
    # ---------------------------------------------------------------------------
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

        # Step 1: global std rescale (preserves kurtosis — uniform scale)
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

        # Step 2: drift correction
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

        # Step 3: measure kurtosis BEFORE any clipping; set sc_inner high
        #         so only extreme outliers (>8-sigma) are soft-clipped.
        final_rets = np.diff(np.log(np.maximum(
            df_result["Close"].values.astype(float), 1e-10
        )))
        topup_mu  = float(np.mean(final_rets))
        topup_std = float(np.std(final_rets))
        if topup_std < 1e-10:
            topup_std = 1e-10
        z_norm = (final_rets - topup_mu) / topup_std

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ek_before = float(stats.kurtosis(z_norm))

        window_ek_vals = [
            row["target_ek"]
            for row in param_log
            if not row.get("_summary") and "target_ek" in row
        ]
        median_target_ek = float(np.median(window_ek_vals)) if window_ek_vals else 6.0

        # v72: inner = max(5, 2*sqrt(target_ek)) — fat tails are NOT clipped
        # e.g. ek=10 → inner=6.3; ek=20 → inner=8.9; ek=3 → inner=5.0
        sc_inner = float(max(5.0, 2.0 * float(np.sqrt(median_target_ek))))
        sc_outer = sc_inner + 3.0

        z_wins = _soft_clip_z(z_norm, inner=sc_inner, outer=sc_outer)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ek_after = float(stats.kurtosis(z_wins))

        if ek_before > 0.5:
            decay_ratio = float(np.clip(ek_after / ek_before, 0.10, 1.0))
            ek_decay_ema = (
                _EK_DECAY_EMA_ALPHA * decay_ratio
                + (1.0 - _EK_DECAY_EMA_ALPHA) * ek_decay_ema
            )

        if verbose:
            print(
                f"[stat] global-pass  ek_before={ek_before:.2f}"
                f"  ek_after={ek_after:.2f}"
                f"  sc_inner={sc_inner:.2f}  sc_outer={sc_outer:.2f}"
                f"  decay={ek_after/max(ek_before,1e-3):.3f}"
                f"  ema={ek_decay_ema:.3f}"
                f"  oversample_next={_EK_OVERSAMPLE/max(ek_decay_ema,0.15):.2f}"
            )

        # Step 4: skew alignment (on z_wins, before rescale)
        z_aligned = _linear_skew_align(z_wins, target_skew=global_skew_target)

        # Step 5: kurtosis topup on z_aligned (in z-score space)
        # target in z-space is median_target_ek (kurtosis is scale-invariant)
        z_toppedUp = _kurtosis_topup(z_aligned, median_target_ek, rng)

        # Step 6: reconstruct returns, pin std precisely
        final_rets_fixed = z_toppedUp * topup_std + topup_mu
        post_std = float(np.std(final_rets_fixed))
        if post_std > 1e-10 and real_global_std > 1e-10:
            # only rescale if std drifted meaningfully (>5%)
            if abs(post_std / real_global_std - 1.0) > 0.05:
                final_rets_fixed = final_rets_fixed / post_std * real_global_std

        new_closes = np.empty(len(df_result))
        new_closes[0] = float(df_result["Close"].iloc[0])
        for i in range(1, len(new_closes)):
            new_closes[i] = new_closes[i - 1] * np.exp(final_rets_fixed[i - 1])
        price_ratio        = new_closes / np.maximum(df_result["Close"].values.astype(float), 1e-10)
        df_result          = df_result.copy()
        df_result["Close"] = new_closes
        df_result["Open"]  = df_result["Open"].values * price_ratio
        df_result["High"]  = df_result["High"].values * price_ratio
        df_result["Low"]   = df_result["Low"].values  * price_ratio

    param_log.append({
        "_summary":  True,
        "n_windows": window_idx,
        "dtw_mean":  float(np.mean(all_dtw))   if all_dtw   else None,
        "pcorr_mean": float(np.mean(all_pcorr)) if all_pcorr else None,
    })

    return df_result, param_log
