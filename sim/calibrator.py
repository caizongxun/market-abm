"""
calibrator.py  v3
=================
AdaptiveCalibrator：replay buffer + XGBoost 校正模型。

v3 fix：
  - load() 加 try/except，捕捉 EOFError / UnpicklingError（崩潰時產生的損壞 pkl），
    自動刪檔並以全新 calibrator 繼續執行。
v2 fix：
  - Ridge fallback 改用 RidgeModel 類別，取代 lambda 閉包，
    解決 pickle.dump 時 "Can't pickle local function" 的問題。

架構
----
  - ReplayBuffer        : 環形 buffer，儲存 (context, action, reward) 三元組
  - CalibAction         : 對 StatParams 各欄位的乘法修正量
  - RidgeModel          : 可 pickle 的 Ridge 預測器（XGBoost 不可用時的 fallback）
  - AdaptiveCalibrator  : 主類，predict() + record() + fit() + save/load

Context 特徵（9 維）
  ret_std, ret_skew_a, ret_df, hurst_target, wick_lambda,
  jump_freq, vol_persistence, acf_lag1, target_ek

Action 輸出（4 維，對應最敏感的 4 個參數）
  d_ret_std, d_hurst, d_target_ek, d_vol_persistence
  -- 解釋為相對 delta（+0.05 = 乘以 1.05）

Reward（純量，越高越好）
  reward = - (0.5*std_err_pct + 0.3*kurt_err/10 + 0.1*hurst_err/0.1 - 0.1*dir_hit)
  bounded to [-5, +1]

持久化
  calibrator.save(path)   # pickle
  calibrator.load(path)   # 就地恢復（損壞檔自動刪除從頭來過）
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

CONTEXT_KEYS: List[str] = [
    "ret_std", "ret_skew_a", "ret_df",
    "hurst_target", "wick_lambda",
    "jump_freq", "vol_persistence", "acf_lag1", "target_ek",
]

ACTION_KEYS: List[str] = [
    "d_ret_std", "d_hurst", "d_target_ek", "d_vol_persistence",
]

ACTION_CLIP: Dict[str, tuple] = {
    "d_ret_std":          (-0.40, 0.40),
    "d_hurst":            (-0.15, 0.15),
    "d_target_ek":        (-0.50, 0.50),
    "d_vol_persistence":  (-0.30, 0.30),
}

ACTION_TARGET: Dict[str, str] = {
    "d_ret_std":          "ret_std",
    "d_hurst":            "hurst_target",
    "d_target_ek":        "target_ek",
    "d_vol_persistence":  "vol_persistence",
}

PARAM_SAFE: Dict[str, tuple] = {
    "ret_std":          (1e-5,  0.20),
    "hurst_target":     (0.30,  0.69),
    "target_ek":        (1.0,   30.0),
    "vol_persistence":  (0.0,   0.85),
}


# ---------------------------------------------------------------------------
# CalibAction
# ---------------------------------------------------------------------------

@dataclass
class CalibAction:
    d_ret_std:         float = 0.0
    d_hurst:           float = 0.0
    d_target_ek:       float = 0.0
    d_vol_persistence: float = 0.0

    def apply(self, params: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(params)
        for action_key, param_key in ACTION_TARGET.items():
            delta = getattr(self, action_key)
            lo, hi = ACTION_CLIP[action_key]
            delta = float(np.clip(delta, lo, hi))
            old_val = float(result[param_key])
            new_val = old_val * (1.0 + delta)
            if param_key in PARAM_SAFE:
                plo, phi = PARAM_SAFE[param_key]
                new_val = float(np.clip(new_val, plo, phi))
            result[param_key] = new_val
        return result

    def to_array(self) -> np.ndarray:
        return np.array([self.d_ret_std, self.d_hurst,
                         self.d_target_ek, self.d_vol_persistence], dtype=float)

    @staticmethod
    def from_array(arr: np.ndarray) -> "CalibAction":
        arr = np.asarray(arr, dtype=float)
        return CalibAction(
            d_ret_std         = float(arr[0]),
            d_hurst           = float(arr[1]),
            d_target_ek       = float(arr[2]),
            d_vol_persistence = float(arr[3]),
        )

    @staticmethod
    def zero() -> "CalibAction":
        return CalibAction(0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self._ctx:    List[np.ndarray] = []
        self._act:    List[np.ndarray] = []
        self._reward: List[float]      = []
        self._ptr: int = 0

    def push(self, ctx: np.ndarray, act: np.ndarray, reward: float) -> None:
        if len(self._ctx) < self.capacity:
            self._ctx.append(ctx)
            self._act.append(act)
            self._reward.append(reward)
        else:
            self._ctx[self._ptr]    = ctx
            self._act[self._ptr]    = act
            self._reward[self._ptr] = reward
            self._ptr = (self._ptr + 1) % self.capacity

    def __len__(self) -> int:
        return len(self._ctx)

    @property
    def contexts(self) -> np.ndarray:
        return np.array(self._ctx, dtype=float)

    @property
    def actions(self) -> np.ndarray:
        return np.array(self._act, dtype=float)

    @property
    def rewards(self) -> np.ndarray:
        return np.array(self._reward, dtype=float)


# ---------------------------------------------------------------------------
# RidgeModel — 可 pickle 的 Ridge fallback
# ---------------------------------------------------------------------------

class RidgeModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        n, d = X.shape
        XtX = X.T @ X + self.alpha * np.eye(d)
        try:
            self.w_ = np.linalg.solve(XtX, X.T @ y)
        except np.linalg.LinAlgError:
            self.w_ = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            return np.zeros(len(np.atleast_2d(X)))
        return np.atleast_2d(X) @ self.w_


# ---------------------------------------------------------------------------
# AdaptiveCalibrator
# ---------------------------------------------------------------------------

class AdaptiveCalibrator:
    """
    跨 trial 持久化的校正器。
    崩潰時產生的損壞 pkl 會被 load() 自動刪除，不影響下次執行。
    """

    def __init__(
        self,
        capacity:         int   = 5000,
        min_train:        int   = 50,
        update_interval:  int   = 20,
        explore_std:      float = 0.03,
        xgb_n_estimators: int   = 80,
        xgb_max_depth:    int   = 4,
        xgb_lr:           float = 0.10,
    ):
        self.min_train       = min_train
        self.update_interval = update_interval
        self.explore_std     = explore_std
        self.xgb_kwargs      = dict(
            n_estimators     = xgb_n_estimators,
            max_depth        = xgb_max_depth,
            learning_rate    = xgb_lr,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            verbosity        = 0,
        )
        self._buffer    = ReplayBuffer(capacity)
        self._models: Optional[List[Any]] = None
        self._n_since_fit: int = 0
        self.n_experiences:  int = 0

    @staticmethod
    def build_context(params: Dict[str, Any]) -> np.ndarray:
        return np.array([float(params[k]) for k in CONTEXT_KEYS], dtype=float)

    @staticmethod
    def _compute_reward(
        std_err_pct: float,
        kurt_err:    float,
        hurst_err:   float,
        dir_hit:     float,
    ) -> float:
        r = -(0.5 * std_err_pct
              + 0.3 * kurt_err / 10.0
              + 0.1 * hurst_err / 0.10
              - 0.1 * dir_hit)
        return float(np.clip(r, -5.0, 1.0))

    def _fit_models(self) -> None:
        if len(self._buffer) < self.min_train:
            return
        X = self._buffer.contexts
        A = self._buffer.actions
        R = self._buffer.rewards
        R_shifted = R - R.min() + 1e-6
        w = R_shifted / R_shifted.sum()
        self._models = []
        for i in range(len(ACTION_KEYS)):
            y        = A[:, i]
            y_target = np.full_like(y, float(np.sum(w * y)))
            y_blend  = (1 - w) * y + w * y_target
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if _HAS_XGB:
                    m = XGBRegressor(**self.xgb_kwargs)
                    m.fit(X, y_blend)
                else:
                    m = RidgeModel(alpha=1.0).fit(X, y_blend)
            self._models.append(m)
        self._n_since_fit = 0

    def predict(self, ctx: np.ndarray) -> CalibAction:
        if self._models is None or len(self._buffer) < self.min_train:
            action = CalibAction.zero()
        else:
            x = ctx.reshape(1, -1)
            deltas = []
            for i, m in enumerate(self._models):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val = float(m.predict(x).flat[0])
                lo, hi = ACTION_CLIP[ACTION_KEYS[i]]
                deltas.append(float(np.clip(val, lo, hi)))
            action = CalibAction.from_array(np.array(deltas))
        if self.explore_std > 0:
            noise = np.random.normal(0, self.explore_std, size=len(ACTION_KEYS))
            a_arr = action.to_array() + noise
            for i, key in enumerate(ACTION_KEYS):
                lo, hi = ACTION_CLIP[key]
                a_arr[i] = float(np.clip(a_arr[i], lo, hi))
            action = CalibAction.from_array(a_arr)
        return action

    def record(
        self,
        context:     np.ndarray,
        action:      CalibAction,
        std_err_pct: float,
        kurt_err:    float,
        hurst_err:   float,
        dir_hit:     float,
    ) -> None:
        reward = self._compute_reward(std_err_pct, kurt_err, hurst_err, dir_hit)
        self._buffer.push(context, action.to_array(), reward)
        self.n_experiences += 1
        self._n_since_fit  += 1
        if (self._n_since_fit >= self.update_interval
                and len(self._buffer) >= self.min_train):
            self._fit_models()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "buffer":          self._buffer,
                "models":          self._models,
                "n_experiences":   self.n_experiences,
                "explore_std":     self.explore_std,
                "min_train":       self.min_train,
                "update_interval": self.update_interval,
                "xgb_kwargs":      self.xgb_kwargs,
            }, f)

    def load(self, path: str) -> None:
        """
        載入 pkl。若檔案損壞（EOFError 或 UnpicklingError），
        自動刪除并以空 calibrator 繼續，不會中斷執行。
        """
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, Exception) as exc:
            print(f"[calib] WARNING: corrupt pkl ({exc}), deleting and starting fresh.")
            try:
                os.remove(path)
            except OSError:
                pass
            return   # 保持目前空狀態

        self._buffer          = state["buffer"]
        self._models          = state.get("models")
        self.n_experiences    = state.get("n_experiences", len(self._buffer))
        self.explore_std      = state.get("explore_std",     self.explore_std)
        self.min_train        = state.get("min_train",        self.min_train)
        self.update_interval  = state.get("update_interval", self.update_interval)
        self.xgb_kwargs       = state.get("xgb_kwargs",      self.xgb_kwargs)
        self._n_since_fit     = 0
        # 若 models 為 None（舊版 pkl 只存 buffer），重新訓練
        if self._models is None and len(self._buffer) >= self.min_train:
            self._fit_models()
