"""
calibrator.py  v1
=================
AdaptiveCalibrator：以壓測本身為訓練來源的線上校正器。

架構
----
每個滾動視窗結束後，記錄一筆：
  context  : 7 維市場特徵（波動率 regime、ek、skew、hurst…）
  action   : 校正後參數調整量（std_scale、ek_adjust、chi2_skip）
  reward   : -（std_err + kurt_err_norm + hurst_err + dir_penalty）

累積 min_train 筆後，用 XGBoost（或 Ridge fallback）擬合
  context → best_action
並在後續視窗預測校正量，逐步減少誤差。

API
---
  cal = AdaptiveCalibrator()
  cal.load("models/calibrator.pkl")      # 可選：載入預訓練

  # 在 rolling_fit_generate 內部每個視窗呼叫：
  action = cal.predict(context)           # 取得本視窗校正量
  ...執行 generate()...
  cal.record(context, action, reward)     # 記錄結果

  cal.save("models/calibrator.pkl")       # 壓測結束後持久化
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Action 結構
# ---------------------------------------------------------------------------

@dataclass
class CalibAction:
    """校正器輸出的參數調整量。"""
    std_scale:  float = 1.0   # 乘以 ret_std（1.0 = 不調）
    ek_adjust:  float = 0.0   # 加到 target_ek
    skew_scale: float = 1.0   # 乘以 ret_skew_a
    hurst_adj:  float = 0.0   # 加到 hurst_target

    def apply(self, params: dict) -> dict:
        """把調整量套用到 StatParams dict，回傳新 dict。"""
        p = dict(params)
        p["ret_std"]      = float(np.clip(p["ret_std"]      * self.std_scale,  1e-6, 1.0))
        p["target_ek"]    = float(np.clip(p["target_ek"]    + self.ek_adjust,   0.5, 60.0))
        p["ret_skew_a"]   = float(np.clip(p["ret_skew_a"]   * self.skew_scale, -10.0, 10.0))
        p["hurst_target"] = float(np.clip(p["hurst_target"] + self.hurst_adj,   0.3,  0.69))
        return p


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

@dataclass
class _Experience:
    context: np.ndarray    # shape (7,)
    action:  np.ndarray    # shape (4,)  [std_scale, ek_adjust, skew_scale, hurst_adj]
    reward:  float


class ReplayBuffer:
    def __init__(self, maxlen: int = 20_000):
        self.maxlen = maxlen
        self._buf: list[_Experience] = []

    def push(self, context: np.ndarray, action: np.ndarray, reward: float) -> None:
        if len(self._buf) >= self.maxlen:
            self._buf.pop(0)
        self._buf.append(_Experience(context.copy(), action.copy(), float(reward)))

    def __len__(self) -> int:
        return len(self._buf)

    def sample_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.array([e.context for e in self._buf], dtype=float)
        A = np.array([e.action  for e in self._buf], dtype=float)
        R = np.array([e.reward  for e in self._buf], dtype=float)
        return X, A, R


# ---------------------------------------------------------------------------
# AdaptiveCalibrator
# ---------------------------------------------------------------------------

class AdaptiveCalibrator:
    """
    線上校正器：壓測每跑一個視窗就學一筆，跨 trial 持續累積知識。

    context 特徵（7 維）
    --------------------
    0  ret_std_norm      : ret_std / 0.015（相對正規化）
    1  target_ek_norm    : target_ek / 10.0
    2  skew_a            : ret_skew_a（原值）
    3  hurst             : hurst_target
    4  vol_persistence   : vol_persistence
    5  acf_lag1          : acf_lag1
    6  df_t_norm         : ret_df / 10.0
    """

    CONTEXT_DIM = 7
    ACTION_DIM  = 4   # std_scale, ek_adjust, skew_scale, hurst_adj

    def __init__(
        self,
        min_train:       int   = 50,
        update_interval: int   = 20,
        explore_std:     float = 0.05,
        use_xgb:         bool  = True,
    ):
        self.min_train       = min_train
        self.update_interval = update_interval
        self.explore_std     = explore_std
        self.use_xgb         = use_xgb

        self.buffer    = ReplayBuffer(maxlen=20_000)
        self._models: list = []          # one model per action dim
        self._trained  = False
        self._step     = 0
        self._rng      = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_context(params: dict) -> np.ndarray:
        return np.array([
            params["ret_std"]        / 0.015,
            params["target_ek"]      / 10.0,
            float(params["ret_skew_a"]),
            float(params["hurst_target"]),
            float(params["vol_persistence"]),
            float(params["acf_lag1"]),
            params["ret_df"]         / 10.0,
        ], dtype=float)

    # ------------------------------------------------------------------
    # Reward builder
    # ------------------------------------------------------------------

    @staticmethod
    def compute_reward(
        std_err_pct: float,
        kurt_err:    float,
        hurst_err:   float,
        dir_hit:     float,
    ) -> float:
        r = -(
            1.0  * min(std_err_pct, 2.0)
            + 0.3 * min(kurt_err / 50.0, 1.0)
            + 2.0 * min(hurst_err * 10, 1.0)
            + 1.5 * max(0.0, 0.52 - dir_hit) * 10
        )
        return float(r)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, context: np.ndarray) -> CalibAction:
        """根據市場 context 預測最佳校正量。未訓練時回傳 no-op。"""
        if not self._trained:
            return self._default_action()

        x = context.reshape(1, -1)
        preds = []
        for model in self._models:
            try:
                p = float(model.predict(x)[0])
            except Exception:
                p = 0.0
            preds.append(p)

        # 加 exploration noise（training 時才用，可設 explore_std=0 關閉）
        noise = self._rng.normal(0, self.explore_std, size=self.ACTION_DIM)
        preds = np.array(preds) + noise

        return CalibAction(
            std_scale  = float(np.clip(preds[0], 0.5, 2.0)),
            ek_adjust  = float(np.clip(preds[1], -10.0, 10.0)),
            skew_scale = float(np.clip(preds[2], 0.5, 2.0)),
            hurst_adj  = float(np.clip(preds[3], -0.2, 0.2)),
        )

    def _default_action(self) -> CalibAction:
        return CalibAction(std_scale=1.0, ek_adjust=0.0, skew_scale=1.0, hurst_adj=0.0)

    # ------------------------------------------------------------------
    # Record & update
    # ------------------------------------------------------------------

    def record(
        self,
        context:     np.ndarray,
        action:      CalibAction,
        std_err_pct: float,
        kurt_err:    float,
        hurst_err:   float,
        dir_hit:     float,
    ) -> None:
        """記錄一筆經驗，並按 update_interval 觸發重新訓練。"""
        reward = self.compute_reward(std_err_pct, kurt_err, hurst_err, dir_hit)
        a_vec  = np.array([
            action.std_scale,
            action.ek_adjust,
            action.skew_scale,
            action.hurst_adj,
        ], dtype=float)
        self.buffer.push(context, a_vec, reward)
        self._step += 1

        if (len(self.buffer) >= self.min_train
                and self._step % self.update_interval == 0):
            self._fit()

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        X, A, R = self.buffer.sample_all()

        # 目標：高 reward 的 action 更值得學
        # 用 reward 做 sample weight（shift 到 0+）
        w = R - R.min() + 1e-3
        w = w / w.sum() * len(w)

        models = []
        for dim in range(self.ACTION_DIM):
            y = A[:, dim]
            model = self._fit_one(X, y, w)
            models.append(model)

        self._models  = models
        self._trained = True

    def _fit_one(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        if self.use_xgb:
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X, y, sample_weight=w)
                return model
            except ImportError:
                pass  # fallback to Ridge

        # Ridge fallback
        return self._ridge_fit(X, y, w)

    @staticmethod
    def _ridge_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray):
        """加權 Ridge 回歸。"""
        class _RidgeModel:
            def __init__(self, coef, intercept):
                self.coef_      = coef
                self.intercept_ = intercept
            def predict(self, X):
                return X @ self.coef_ + self.intercept_

        W     = np.diag(w)
        XtW   = X.T @ W
        alpha = 1.0
        A_mat = XtW @ X + alpha * np.eye(X.shape[1])
        b_vec = XtW @ y
        try:
            coef = np.linalg.solve(A_mat, b_vec)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
        intercept = float(np.mean(y) - coef @ np.mean(X, axis=0))
        return _RidgeModel(coef, intercept)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "buffer":    self.buffer,
                "models":    self._models,
                "trained":   self._trained,
                "step":      self._step,
                "min_train": self.min_train,
                "update_interval": self.update_interval,
                "explore_std":     self.explore_std,
                "use_xgb":         self.use_xgb,
            }, f)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.buffer          = d["buffer"]
        self._models         = d["models"]
        self._trained        = d["trained"]
        self._step           = d["step"]
        self.min_train       = d.get("min_train",       self.min_train)
        self.update_interval = d.get("update_interval", self.update_interval)
        self.explore_std     = d.get("explore_std",     self.explore_std)
        self.use_xgb         = d.get("use_xgb",         self.use_xgb)

    @property
    def n_experiences(self) -> int:
        return len(self.buffer)

    @property
    def is_trained(self) -> bool:
        return self._trained
