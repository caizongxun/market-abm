"""
adaptive_params.py  v26
=======================
Feature Extractor + OnlineRidgePredictor

v26 變更：
  TARGET_NAMES 新增 target_ek 為第 4 個預測目標
  Ridge 學習市場何時尾部特別肥，自適應調整 p_t 的上游參數
  predict_and_blend() blend target_ek，clip 到 (1.0, 25.0)

設計原則：
- 完全獨立，不改動 stat_process.py 的 fit() 和 generate()
- 前 min_train 個 window 純觀察，不介入參數
- 之後用 blend_w 做加權混合，最大只信 50%，保留 fit() 穩健性
- 所有 features 使用相對值（比例、rank），跨市場/跨時間段通用

Features（7個）：
  std_rank          當前 window ret_std / 歷史最大 ret_std
  vol_trend         後半 30 bar std / 前半 30 bar std
  price_vs_high20   fit_end 收盤 / 前 20 bar rolling max（恐懼指標）
  price_vs_low20    fit_end 收盤 / 前 20 bar rolling min（貪婪指標）
  hurst_lag         前一個 window 的 hurst_target
  skew_sign_streak  連續幾個 window skew_a 同號（正=持續正skew，負=持續負skew）
  consec_direction  fit_end 前連續上漲(+)或下跌(-)的 bar 數

Targets（4個）：
  ret_std, ret_skew_a, hurst_target, target_ek
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sim.stat_process import StatParams


# ---------------------------------------------------------------------------
# Feature names（有序，供外部診斷用）
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "std_rank",
    "vol_trend",
    "price_vs_high20",
    "price_vs_low20",
    "hurst_lag",
    "skew_sign_streak",
    "consec_direction",
]

TARGET_NAMES = ["ret_std", "ret_skew_a", "hurst_target", "target_ek"]  # v26: +target_ek


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

def extract_features(
    df_real:     pd.DataFrame,
    fit_end:     int,
    past_params: list["StatParams"],
) -> np.ndarray:
    """
    Parameters
    ----------
    df_real     : 完整的真實 OHLC DataFrame
    fit_end     : 當前 window 的結束 index（exclusive）
    past_params : 已累積的所有 StatParams（包含剛 fit 完的當前 window）

    Returns
    -------
    np.ndarray shape (7,)  — 對應 FEATURE_NAMES 的順序
    """
    closes = df_real["Close"].values.astype(float)
    end_idx = min(fit_end, len(closes))

    # --- std_rank ----------------------------------------------------------
    if len(past_params) >= 2:
        hist_stds = [p["ret_std"] for p in past_params[:-1]]
        max_std   = max(hist_stds) if hist_stds else past_params[-1]["ret_std"]
        std_rank  = float(past_params[-1]["ret_std"] / max(max_std, 1e-10))
    else:
        std_rank = 1.0
    std_rank = float(np.clip(std_rank, 0.1, 3.0))

    # --- vol_trend ---------------------------------------------------------
    lookback = 60
    start_idx = max(0, end_idx - lookback)
    window_closes = closes[start_idx:end_idx]
    if len(window_closes) >= 20:
        log_rets = np.diff(np.log(np.maximum(window_closes, 1e-10)))
        mid      = len(log_rets) // 2
        std_back = float(np.std(log_rets[mid:])) if mid < len(log_rets) else 1e-10
        std_fore = float(np.std(log_rets[:mid]))  if mid > 0              else 1e-10
        vol_trend = float(np.clip(std_back / max(std_fore, 1e-10), 0.2, 5.0))
    else:
        vol_trend = 1.0

    # --- price_vs_high20 / price_vs_low20 ----------------------------------
    lookback20 = min(20, end_idx)
    recent_closes = closes[end_idx - lookback20:end_idx]
    last_close    = closes[end_idx - 1] if end_idx > 0 else 1.0
    if len(recent_closes) >= 2:
        roll_max = float(np.max(recent_closes))
        roll_min = float(np.min(recent_closes))
        price_vs_high20 = float(np.clip(last_close / max(roll_max, 1e-10), 0.5, 1.5))
        price_vs_low20  = float(np.clip(last_close / max(roll_min, 1e-10), 0.5, 3.0))
    else:
        price_vs_high20 = 1.0
        price_vs_low20  = 1.0

    # --- hurst_lag ---------------------------------------------------------
    if len(past_params) >= 2:
        hurst_lag = float(past_params[-2]["hurst_target"])
    else:
        hurst_lag = 0.5

    # --- skew_sign_streak --------------------------------------------------
    if len(past_params) >= 2:
        streak = 1
        sign0  = np.sign(past_params[-1]["ret_skew_a"])
        for p in reversed(past_params[:-1]):
            if np.sign(p["ret_skew_a"]) == sign0:
                streak += 1
            else:
                break
        skew_sign_streak = float(sign0 * streak)
    else:
        skew_sign_streak = 0.0
    skew_sign_streak = float(np.clip(skew_sign_streak, -10.0, 10.0))

    # --- consec_direction --------------------------------------------------
    consec = 0
    if end_idx >= 2:
        direction = np.sign(closes[end_idx - 1] - closes[end_idx - 2])
        for i in range(end_idx - 2, max(end_idx - 21, 0), -1):
            d = np.sign(closes[i] - closes[i - 1]) if i > 0 else 0
            if d == direction and direction != 0:
                consec += 1
            else:
                break
        consec = int(direction * consec)
    consec_direction = float(np.clip(consec, -20, 20))

    return np.array([
        std_rank,
        vol_trend,
        price_vs_high20,
        price_vs_low20,
        hurst_lag,
        skew_sign_streak,
        consec_direction,
    ], dtype=float)


# ---------------------------------------------------------------------------
# OnlineRidgePredictor
# ---------------------------------------------------------------------------

class OnlineRidgePredictor:
    """
    Online Ridge Regression predictor。

    v26: targets 從 3 個擴充到 4 個，新增 target_ek

    - observe()             : 每個 window 跑完後記錄 (features, targets)
    - predict_and_blend()  : 用 Ridge 預測，與 fit() 結果加權混合
    - blend_w 從 0 線性增長到 max_blend（預設 0.5），越跑越信任 predictor

    Parameters
    ----------
    min_train  : 開始預測前需要的最小訓練樣本數（預設 10）
    max_blend  : predictor 預測值的最大混合權重（預設 0.5）
    alpha      : Ridge 正則化係數（預設 1.0）
    verbose    : 是否印出 predictor 資訊
    """

    def __init__(
        self,
        min_train: int   = 10,
        max_blend: float = 0.5,
        alpha:     float = 1.0,
        verbose:   bool  = True,
    ):
        self.min_train  = min_train
        self.max_blend  = max_blend
        self.alpha      = alpha
        self.verbose    = verbose

        self._X: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._models: dict        = {}

    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return len(self._X)

    @property
    def ready(self) -> bool:
        return self.n_samples >= self.min_train

    # ------------------------------------------------------------------
    def observe(
        self,
        features: np.ndarray,
        params:   "StatParams",
    ) -> None:
        """記錄當前 window 的 features 和 fit() 產出的 params 作為訓練資料。"""
        targets = np.array([
            params["ret_std"],
            params["ret_skew_a"],
            params["hurst_target"],
            params["target_ek"],      # v26: 新增
        ], dtype=float)
        self._X.append(features.copy())
        self._y.append(targets.copy())
        self._refit()

    # ------------------------------------------------------------------
    def _refit(self) -> None:
        """用目前所有樣本重新 fit Ridge（樣本不多時很快）。"""
        if not self.ready:
            return
        try:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return

        X = np.array(self._X)   # (n, 7)
        y = np.array(self._y)   # (n, 4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X)
            model  = Ridge(alpha=self.alpha)
            model.fit(X_sc, y)

        self._models["ridge"]  = model
        self._models["scaler"] = scaler

    # ------------------------------------------------------------------
    def predict_and_blend(
        self,
        features: np.ndarray,
        params:   "StatParams",
    ) -> "StatParams":
        """
        用 Ridge 預測四個目標參數，與 fit() 結果加權混合後回傳新的 StatParams。

        blend_w 隨樣本數線性增長：
          blend_w = max_blend * (n_samples - min_train) / max(min_train, 1)
          上限 max_blend = 0.5
        """
        if not self.ready or "ridge" not in self._models:
            return params

        blend_w = float(np.clip(
            self.max_blend * (self.n_samples - self.min_train) / max(self.min_train, 1),
            0.0,
            self.max_blend,
        ))

        try:
            X_sc   = self._models["scaler"].transform(features.reshape(1, -1))
            pred   = self._models["ridge"].predict(X_sc)[0]   # shape (4,)
        except Exception:
            return params

        pred_std   = float(pred[0])
        pred_skew  = float(pred[1])
        pred_hurst = float(pred[2])
        pred_ek    = float(pred[3])   # v26: 新增

        # 安全 clip
        pred_std   = float(np.clip(pred_std,   1e-4, 0.15))
        pred_skew  = float(np.clip(pred_skew, -10.0, 10.0))
        pred_hurst = float(np.clip(pred_hurst,  0.3,  0.69))
        pred_ek    = float(np.clip(pred_ek,     1.0,  25.0))  # v26

        # 加權混合
        new_std   = (1 - blend_w) * params["ret_std"]      + blend_w * pred_std
        new_skew  = (1 - blend_w) * params["ret_skew_a"]   + blend_w * pred_skew
        new_hurst = (1 - blend_w) * params["hurst_target"] + blend_w * pred_hurst
        new_ek    = (1 - blend_w) * params["target_ek"]    + blend_w * pred_ek  # v26

        if self.verbose:
            print(
                f"         [adapt] blend_w={blend_w:.2f}  "
                f"std {params['ret_std']:.4f}->{new_std:.4f}  "
                f"skew {params['ret_skew_a']:+.3f}->{new_skew:+.3f}  "
                f"hurst {params['hurst_target']:.3f}->{new_hurst:.3f}  "
                f"tgt_ek {params['target_ek']:.2f}->{new_ek:.2f}  "
                f"(n={self.n_samples})"
            )

        new_params: StatParams = dict(params)  # type: ignore[assignment]
        new_params["ret_std"]      = new_std
        new_params["ret_skew_a"]   = new_skew
        new_params["hurst_target"] = new_hurst
        new_params["target_ek"]    = new_ek    # v26
        return new_params  # type: ignore[return-value]
