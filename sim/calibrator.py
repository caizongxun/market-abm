"""
calibrator.py  v4
=================
AdaptiveCalibrator：replay buffer + XGBoost 校正模型。

v4 patch（兩項）：
  Patch-2  _compute_reward：kurt_err 改用 log1p 壓縮 + 更高權重
           舊: -(0.5*std_err_pct + 0.3*kurt_err/10 + 0.1*hurst_err/0.1 - 0.1*dir_hit)
           新: -(0.3*std_err_pct + 0.4*log1p(kurt_err)/3 + 0.2*hurst_err/0.1 - 0.1*dir_hit)
  Patch-3  build_context：target_ek 改為 log1p 正規化，
           解決 ret_std(~0.01) vs target_ek(~300) 量級差 4 個 order 的問題。

v3 fix：load() 加 corrupt pkl 守衛
v2 fix：RidgeModel 取代 lambda 閉包
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
    "target_ek":        (1.0,   15.0),   # Patch-1 同步上限
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
# RidgeModel
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
    跨 trial 持久化校正器。
    Patch-2: reward 強化 kurtosis 懲罰。
    Patch-3: context 用 log1p 正規化 target_ek。
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
        """
        Patch-3: target_ek 改為 log1p 正規化，壓縮量級差異。
        其餘 8 個特徵維持原值。
        context 維度仍為 9（介面不變）。
        """
        return np.array([
            float(params["ret_std"]),
            float(params["ret_skew_a"]),
            float(params["ret_df"]),
            float(params["hurst_target"]),
            float(params["wick_lambda"]),
            float(params["jump_freq"]),
            float(params["vol_persistence"]),
            float(params["acf_lag1"]),
            float(np.log1p(abs(params["target_ek"]))),  # Patch-3: log1p
        ], dtype=float)

    @staticmethod
    def _compute_reward(
        std_err_pct: float,
        kurt_err:    float,
        hurst_err:   float,
        dir_hit:     float,
    ) -> float:
        """
        Patch-2: kurt_err 改用 log1p 壓縮 + 提高權重。
        舊: -(0.5*std + 0.03*kurt + 0.1*hurst - 0.1*dir)
        新: -(0.3*std + 0.4*log1p(kurt)/3 + 0.2*hurst - 0.1*dir)
        """
        r = -(0.3 * std_err_pct
              + 0.4 * float(np.log1p(kurt_err)) / 3.0
              + 0.2 * hurst_err / 0.10
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
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, Exception) as exc:
            print(f"[calib] WARNING: corrupt pkl ({exc}), deleting and starting fresh.")
            try:
                os.remove(path)
            except OSError:
                pass
            return
        self._buffer          = state["buffer"]
        self._models          = state.get("models")
        self.n_experiences    = state.get("n_experiences", len(self._buffer))
        self.explore_std      = state.get("explore_std",     self.explore_std)
        self.min_train        = state.get("min_train",        self.min_train)
        self.update_interval  = state.get("update_interval", self.update_interval)
        self.xgb_kwargs       = state.get("xgb_kwargs",      self.xgb_kwargs)
        self._n_since_fit     = 0
        if self._models is None and len(self._buffer) >= self.min_train:
            self._fit_models()
