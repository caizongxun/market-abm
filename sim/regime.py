"""
regime.py
=========
RegimeCalibrator: 每隔 `step` 根 K 棒，對前 `lookback` 根
做一次小 grid search，用 EMA 平滑參數，再繼續模擬。

設計原則
--------
- grid 比 ParamCalibrator 小（3x3x3x3 = 81 組）
- EMA 平滑（alpha=0.4）：新估値影響 40%，歷史殘留 60%
  - alpha 大 → 跟得快但抖；alpha 小 → 穩定但反應慢
  - 0.4 是中間偏快的設定，避免參數斷層
- loss 加入 p_up penalty + vol_err，同時控制方向偏差和波動率
- std_ratio 超過 1.5 的組合 loss 直接加倍淤汰不合理高波動

EMA 公式
---------
  param_ema[t] = alpha * param_new[t] + (1 - alpha) * param_ema[t-1]

loss 公式
---------
  loss = w_hurst * |hurst_err|
       + w_dir   * (1 - dir_hit)
       + w_kurt  * |kurt_err| / (|real_kurt| + 1)
       + w_vol   * |std_err|  / (real_std + 1e-8)
       + w_pup   * p_up penalty

  若 sim_std / real_std > std_ratio_cap (1.5) → loss *= 2.0

  波動率目標用 lookback 窗口的 real_std，而非 step（20 根），
  避免短序列估算不穩定。

使用方式
--------
  from sim.regime import RegimeCalibrator
  cal = RegimeCalibrator(lookback=60, step=20, n_sims=10, ema_alpha=0.4)
  df_result, param_log = cal.run(df_all)
"""
from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd

from .simulation import run_simulation
from .metrics import compare, hurst_exponent, log_returns

# ---------------------------------------------------------------------------
# Small grid for per-window search
# 3x3x3x3 = 81 combos, ~10 sims each = 810 paths per window
# intra_noise_scale 上界擴展到 1.5，讓 noise 不再一直貿頂
# ---------------------------------------------------------------------------
ROLLING_GRID: dict[str, list] = {
    "impact_coeff":      [0.0008, 0.0015, 0.0025],
    "momentum_scale":    [0.5,    1.0,    2.0   ],
    "decay":             [0.90,   0.93,   0.97  ],
    "intra_noise_scale": [0.7,    1.0,    1.5   ],  # 上界: 1.0 -> 1.5
}


class RegimeCalibrator:
    """
    Parameters
    ----------
    lookback : int
        用前幾根 K 棒估算當前 regime 的參數。預設 60。
    step : int
        每隔幾根 K 棒重新校準一次。預設 20。
    n_sims : int
        每組參數跑幾條路徑取 median。預設 10（快速）。
    warmup_bars : int
        run_simulation 的 warmup context。預設 60。
    ema_alpha : float
        EMA 平滑係數。建議範圍 0.3–0.6。
    grid : dict
        參數搜尋網格，預設 ROLLING_GRID（3x3x3x3 = 81）。
    w_hurst, w_dir, w_kurt, w_vol, w_pup : float
        loss 各項權重。
    p_up_lo, p_up_hi : float
        p_up 合理範圍，超出會加 penalty。
    std_ratio_cap : float
        sim_std / real_std 超過此値時 loss 乘以 2.0。預設 1.5。
    verbose : bool
        是否印每個 window 的搜尋結果。
    """

    def __init__(
        self,
        lookback: int = 60,
        step: int = 20,
        n_sims: int = 10,
        warmup_bars: int = 60,
        ema_alpha: float = 0.4,
        grid: dict[str, list] | None = None,
        w_hurst: float = 1.0,
        w_dir:   float = 1.0,
        w_kurt:  float = 0.5,
        w_vol:   float = 1.5,
        w_pup:   float = 1.5,
        p_up_lo: float = 0.25,
        p_up_hi: float = 0.60,
        std_ratio_cap: float = 1.5,
        verbose: bool = True,
    ):
        self.lookback      = lookback
        self.step          = step
        self.n_sims        = n_sims
        self.warmup_bars   = warmup_bars
        self.ema_alpha     = ema_alpha
        self.grid          = grid if grid is not None else ROLLING_GRID
        self.w_hurst       = w_hurst
        self.w_dir         = w_dir
        self.w_kurt        = w_kurt
        self.w_vol         = w_vol
        self.w_pup         = w_pup
        self.p_up_lo       = p_up_lo
        self.p_up_hi       = p_up_hi
        self.std_ratio_cap = std_ratio_cap
        self.verbose       = verbose

    # ------------------------------------------------------------------
    def _loss(
        self,
        metrics: dict,
        real_hurst: float,
        real_kurtosis: float,
        real_std: float,
        p_up: float,
    ) -> float:
        sim = metrics["sim"]
        sim_std   = sim["std"]
        hurst_err = abs(sim["hurst"] - real_hurst)
        kurt_err  = abs(sim["kurtosis"] - real_kurtosis) / (abs(real_kurtosis) + 1.0)
        vol_err   = abs(sim_std - real_std) / (real_std + 1e-8)
        dir_term  = 1.0 - metrics["direction_hit_rate"]
        pup_pen   = max(0.0, p_up - self.p_up_hi) + max(0.0, self.p_up_lo - p_up)

        loss = (
            self.w_hurst * hurst_err
            + self.w_dir  * dir_term
            + self.w_kurt * kurt_err
            + self.w_vol  * vol_err
            + self.w_pup  * pup_pen
        )

        # std_ratio penalty: 超高波動的組合直接淤汰
        if real_std > 1e-8 and (sim_std / real_std) > self.std_ratio_cap:
            loss *= 2.0

        return loss

    # ------------------------------------------------------------------
    def _search_window(
        self,
        df_train: pd.DataFrame,
        df_target: pd.DataFrame,
        seed: int = 0,
    ) -> dict[str, Any]:
        """
        在 df_target（長度 = step）上做小 grid search。
        波動率目標用 df_train（lookback 窗口）估算，不用 20 根的 df_target。
        """
        # --- targets from LOOKBACK window (stable) ---
        train_rets = log_returns(df_train["Close"].values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            real_hurst = hurst_exponent(train_rets)
        from scipy import stats as scipy_stats
        real_kurtosis = float(scipy_stats.kurtosis(train_rets))
        real_std      = float(np.std(train_rets))

        keys   = list(self.grid.keys())
        combos = list(itertools.product(*[self.grid[k] for k in keys]))

        best_loss   = float("inf")
        best_params = dict(zip(keys, combos[0]))

        for combo in combos:
            params = dict(zip(keys, combo))
            paths  = []
            for i in range(self.n_sims):
                _, df_sim = run_simulation(
                    df_real=df_train,
                    sim_bars=len(df_target),
                    warmup_bars=self.warmup_bars,
                    impact_coeff=params["impact_coeff"],
                    momentum_scale=params["momentum_scale"],
                    bias_decay=params["decay"],
                    intra_noise_scale=params["intra_noise_scale"],
                    use_momentum_init=True,
                    auto_drift=True,
                    seed=seed + i,
                )
                paths.append(df_sim)

            close_mat  = np.vstack([p["Close"].values for p in paths])
            med_close  = np.median(close_mat, axis=0)
            df_med     = paths[0].copy()
            df_med["Close"] = med_close
            m = compare(df_train, df_med, print_report=False)

            final_rets = (close_mat[:, -1] - close_mat[:, 0]) / close_mat[:, 0]
            p_up = float((final_rets > 0).mean())

            loss = self._loss(m, real_hurst, real_kurtosis, real_std, p_up)
            if loss < best_loss:
                best_loss   = loss
                best_params = {**params, "loss": loss, "p_up": p_up,
                               "hurst_sim": m["sim"]["hurst"],
                               "real_hurst": real_hurst,
                               "real_std": real_std,
                               "sim_std": m["sim"]["std"]}

        return best_params

    # ------------------------------------------------------------------
    def _ema_update(
        self,
        current: dict[str, float],
        new: dict[str, float],
        alpha: float,
    ) -> dict[str, float]:
        keys = ["impact_coeff", "momentum_scale", "decay", "intra_noise_scale"]
        return {k: alpha * new[k] + (1 - alpha) * current[k] for k in keys}

    # ------------------------------------------------------------------
    def run(
        self,
        df_all: pd.DataFrame,
        seed: int = 0,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """
        Rolling calibration 主迴圈。

        Parameters
        ----------
        df_all : pd.DataFrame
            完整 OHLCV（含 warmup + 模擬段）。
        seed : int
            亂數種子起點。

        Returns
        -------
        df_result : pd.DataFrame
            每根模擬 bar 的 OHLCV（拼接所有 window）。
        param_log : list[dict]
            每個 window 的參數記錄。
        """
        n_total   = len(df_all)
        sim_start = self.lookback

        if n_total < sim_start + self.step:
            raise ValueError(
                f"df_all 長度 ({n_total}) 不足 lookback ({self.lookback}) + step ({self.step})"
            )

        current_params = {
            "impact_coeff":      float(np.median(self.grid["impact_coeff"])),
            "momentum_scale":    float(np.median(self.grid["momentum_scale"])),
            "decay":             float(np.median(self.grid["decay"])),
            "intra_noise_scale": float(np.median(self.grid["intra_noise_scale"])),
        }

        all_bars:  list[pd.DataFrame] = []
        param_log: list[dict]         = []
        window_idx = 0

        pos = sim_start
        n_combos = len(list(itertools.product(*self.grid.values())))
        while pos + self.step <= n_total:
            df_train  = df_all.iloc[max(0, pos - self.lookback): pos].copy().reset_index(drop=True)
            df_target = df_all.iloc[pos: pos + self.step].copy().reset_index(drop=True)

            best = self._search_window(
                df_train  = df_train,
                df_target = df_target,
                seed      = seed + window_idx * self.n_sims * n_combos,
            )

            smoothed = self._ema_update(current_params, best, self.ema_alpha)
            current_params = smoothed

            if self.verbose:
                print(
                    f"[regime] window {window_idx+1:>3}  "
                    f"bars [{pos}:{pos+self.step}]  "
                    f"impact={smoothed['impact_coeff']:.4f}  "
                    f"scale={smoothed['momentum_scale']:.2f}  "
                    f"noise={smoothed['intra_noise_scale']:.2f}  "
                    f"decay={smoothed['decay']:.3f}  "
                    f"loss={best.get('loss', float('nan')):.4f}  "
                    f"std sim/real={best.get('sim_std', 0)*100:.3f}%/{best.get('real_std', 0)*100:.3f}%"
                )

            param_log.append({
                "window":            window_idx,
                "bar_start":         pos,
                "bar_end":           pos + self.step,
                "impact_coeff":      smoothed["impact_coeff"],
                "momentum_scale":    smoothed["momentum_scale"],
                "intra_noise_scale": smoothed["intra_noise_scale"],
                "decay":             smoothed["decay"],
                "loss":              best.get("loss",       float("nan")),
                "p_up":              best.get("p_up",       float("nan")),
                "hurst_sim":         best.get("hurst_sim",  float("nan")),
                "real_hurst":        best.get("real_hurst", float("nan")),
                "sim_std":           best.get("sim_std",    float("nan")),
                "real_std":          best.get("real_std",   float("nan")),
            })

            _, df_window_sim = run_simulation(
                df_real=df_train,
                sim_bars=self.step,
                warmup_bars=min(self.warmup_bars, len(df_train)),
                impact_coeff=smoothed["impact_coeff"],
                momentum_scale=smoothed["momentum_scale"],
                bias_decay=smoothed["decay"],
                intra_noise_scale=smoothed["intra_noise_scale"],
                use_momentum_init=True,
                auto_drift=True,
                seed=seed + window_idx,
            )
            all_bars.append(df_window_sim)

            pos        += self.step
            window_idx += 1

        df_result = pd.concat(all_bars, ignore_index=True)
        return df_result, param_log
