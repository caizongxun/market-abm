"""
regime.py
=========
RegimeCalibrator: 每隔 `step` 根 K 棒，對前 `lookback` 根
做一次小 grid search，用 EMA 平滑參數，再繼續模擬。

設計原則
--------
- grid：3x3x3x4 = 108 組（intra_noise_scale 加入 0.4 下界）
- EMA 平滑（alpha=0.4）：新估值影響 40%，歷史殘留 60%
  - alpha 大 → 跟得快但抖；alpha 小 → 穩定但反應慢
  - 0.4 是中間偏快的設定，避免參數斷層
- loss 加入 p_up penalty + vol_err + abs_vol_pen
- std_ratio 對稱 penalty：sim_std/real_std 超出 (cap, 1/cap) 區間時 loss *= 2.0
- impact_coeff grid 收縮到 [0.0005, 0.0010, 0.0015]，抑制 sim_std 偏高

Grid 更新說明（intra_noise_scale）
-----------------------------------
去掉 _t_draw() 的 clip 後，t(4) 尾巴變重，calibrator 需要能把
intra_noise_scale 調到 0.4 以下才能把 std 壓回 real_std 水位。
原本下界 0.7 在無截斷環境下已不夠低，擴展為 [0.4, 0.7, 1.0, 1.5]。
grid 從 81 組增加到 108 組（+33%），每次執行時間等比增加。

Window 接縫對齊（價格平移）
--------------------------
用對數報酬還原法把整個 window 的模擬路徑平移到 anchor_close：

  scale = anchor_close / df_window_sim["Open"].iloc[0]
  df_window_sim[["Open","High","Low","Close"]] *= scale

這保留內部對數報酬的分佈（std / hurst / kurtosis 不變），
同時消除跨 window 的價格跳層漂移，讓 global std 從結構上接近 real_std。

EMA 公式
---------
  param_ema[t] = alpha * param_new[t] + (1 - alpha) * param_ema[t-1]

loss 公式
---------
  loss = w_hurst * |hurst_err|
       + w_dir   * (1 - dir_hit)
       + w_kurt  * |kurt_err| / (|real_kurt| + 1)
       + w_vol   * |std_err|  / (real_std + 1e-8)
       + w_vol   * max(0, sim_std - 2*real_std) / (real_std + 1e-8)  <- abs_vol_pen
       + w_pup   * p_up penalty

  ratio = sim_std / real_std
  若 ratio > std_ratio_cap (1.5) 或 ratio < 1/std_ratio_cap (0.667) → loss *= 2.0

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
# 3x3x3x4 = 108 combos, ~10 sims each = 1080 paths per window
# intra_noise_scale lower bound extended from 0.7 → 0.4 to accommodate
# the heavier t(4) tails after removing the hard clip in _t_draw().
# ---------------------------------------------------------------------------
ROLLING_GRID: dict[str, list] = {
    "impact_coeff":      [0.0005, 0.0010, 0.0015],
    "momentum_scale":    [0.5,    1.0,    2.0   ],
    "decay":             [0.90,   0.93,   0.97  ],
    "intra_noise_scale": [0.4,    0.7,    1.0,   1.5],
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
        參數搜尋網格，預設 ROLLING_GRID（3x3x3x4 = 108）。
    w_hurst, w_dir, w_kurt, w_vol, w_pup : float
        loss 各項權重。
    p_up_lo, p_up_hi : float
        p_up 合理範圍，超出會加 penalty。
    std_ratio_cap : float
        sim_std/real_std 超出 (cap, 1/cap) 區間時 loss 乘以 2.0。預設 1.5。
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
        w_vol:   float = 3.0,
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

        abs_vol_pen = max(0.0, sim_std - 2.0 * real_std) / (real_std + 1e-8)

        loss = (
            self.w_hurst * hurst_err
            + self.w_dir  * dir_term
            + self.w_kurt * kurt_err
            + self.w_vol  * vol_err
            + self.w_vol  * abs_vol_pen
            + self.w_pup  * pup_pen
        )

        if real_std > 1e-8:
            ratio = sim_std / real_std
            if ratio > self.std_ratio_cap or ratio < (1.0 / self.std_ratio_cap):
                loss *= 2.0

        return loss

    # ------------------------------------------------------------------
    def _search_window(
        self,
        df_train: pd.DataFrame,
        df_target: pd.DataFrame,
        seed: int = 0,
    ) -> dict[str, Any]:
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
    @staticmethod
    def _rescale_window(
        df_window: pd.DataFrame,
        anchor_close: float,
    ) -> pd.DataFrame:
        """
        把整個 window 的 OHLC 乘上同一個 scale，
        讓第一根 bar 的 Open == anchor_close。

        用對數報酬還原法：內部每根 bar 的 log-return 完全不變，
        只是整體水位平移，消除跨 window 的價格跳層。
        """
        df = df_window.copy()
        first_open = float(df["Open"].iloc[0])
        if first_open <= 0:
            return df
        scale = anchor_close / first_open
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col] * scale
        return df

    # ------------------------------------------------------------------
    def run(
        self,
        df_all: pd.DataFrame,
        seed: int = 0,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """
        Rolling calibration 主迴圈。

        每個 window 的最終模擬路徑會經過 _rescale_window() 平移，
        確保第一根 bar 的 Open 等於前一個 window 的模擬收盤價，
        消除跨 window 跳層對 global std 的新增展。

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

            # -------------------------------------------------------
            # 接縫對齊：取前一個 window 最後一根模擬收盤價
            # window_idx == 0 時用 df_train 最後一根真實 Close
            # -------------------------------------------------------
            if window_idx == 0:
                anchor_close = float(df_train["Close"].iloc[-1])
            else:
                anchor_close = float(all_bars[-1]["Close"].iloc[-1])

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

            # 對數報酬平移：內部 log-return 不變，只消除跨 window 價格跳層
            df_window_sim = self._rescale_window(df_window_sim, anchor_close)

            all_bars.append(df_window_sim)

            pos        += self.step
            window_idx += 1

        df_result = pd.concat(all_bars, ignore_index=True)
        return df_result, param_log
