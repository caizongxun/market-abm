"""
calibrator.py  v1
=================
AdaptiveCalibrator：replay buffer + XGBoost 校正模型。

架構
----
  - ReplayBuffer        : 環形 buffer，儲存 (context, action, reward) 三元組
  - CalibAction         : 對 StatParams 各欄位的加法/乘法修正量
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
  calibrator.load(path)   # 就地恢復
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# XGBoost 為可選依賴；若不存在則退化到 Ridge
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

# action 合法範圍（相對 delta）
ACTION_CLIP: Dict[str, tuple] = {
    "d_ret_std":          (-0.40, 0.40),
    "d_hurst":            (-0.15, 0.15),
    "d_target_ek":        (-0.50, 0.50),
    "d_vol_persistence":  (-0.30, 0.30),
}

# action 對應 StatParams 的欄位名稱
ACTION_TARGET: Dict[str, str] = {
    "d_ret_std":          "ret_std",
    "d_hurst":            "hurst_target",
    "d_target_ek":        "target_ek",
    "d_vol_persistence":  "vol_persistence",
}

# StatParams 欄位的安全範圍（apply 後 clip）
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
    """校正向量，每個分量為相對 delta（d_x 表示乘以 (1+d_x)）。"""
    d_ret_std:         float = 0.0
    d_hurst:           float = 0.0
    d_target_ek:       float = 0.0
    d_vol_persistence: float = 0.0

    def apply(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """就地修改並回傳 params dict。"""
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
    """固定容量的環形 buffer，存放 (context_vec, action_vec, reward) 三元組。"""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self._ctx:    List[np.ndarray] = []
        self._act:    List[np.ndarray] = []
        self._reward: List[float]      = []
        self._ptr: int = 0
        self._full: bool = False

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
            self._full = True

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
# Ridge fallback
# ---------------------------------------------------------------------------

def _ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred:  np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """最小二乘 Ridge，回傳 X_pred 的預測值。"""
    n, d = X_train.shape
    XtX = X_train.T @ X_train + alpha * np.eye(d)
    try:
        w = np.linalg.solve(XtX, X_train.T @ y_train)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(XtX, X_train.T @ y_train, rcond=None)[0]
    return X_pred @ w


# ---------------------------------------------------------------------------
# AdaptiveCalibrator
# ---------------------------------------------------------------------------

class AdaptiveCalibrator:
    """
    跨 trial 持久化的校正器。

    工作流程
    --------
    1. bench_generalize / rolling_fit_generate 呼叫 predict(ctx) 取得 CalibAction
    2. 用 action.apply(params) 修正 StatParams
    3. 產生模擬、計算誤差指標
    4. 呼叫 record(ctx, action, **err_metrics) 將經驗存入 buffer
    5. 每 update_interval 筆自動重新訓練 XGBoost 模型
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
        self.explore_std     = explore_std   # Gaussian exploration noise
        self.xgb_kwargs      = dict(
            n_estimators = xgb_n_estimators,
            max_depth    = xgb_max_depth,
            learning_rate= xgb_lr,
            subsample    = 0.8,
            colsample_bytree=0.8,
            verbosity    = 0,
        )

        self._buffer     = ReplayBuffer(capacity)
        self._models: Optional[List[Any]] = None   # list of 4 regressors (one per action dim)
        self._n_since_fit: int = 0
        self.n_experiences: int = 0

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_context(params: Dict[str, Any]) -> np.ndarray:
        """從 StatParams dict 抽出 9 維 context 向量。"""
        return np.array([float(params[k]) for k in CONTEXT_KEYS], dtype=float)

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Fit internal models
    # ------------------------------------------------------------------

    def _fit_models(self) -> None:
        if len(self._buffer) < self.min_train:
            return

        X = self._buffer.contexts   # (N, 9)
        A = self._buffer.actions    # (N, 4)
        R = self._buffer.rewards    # (N,)

        # 目標：對每個 action 維度，學習 context -> best_action_dim
        # 簡化監督：用 reward 加權的 A 值作為目標
        # target_i = sum(R * A[:,i]) / (sum(R) + eps)  -- reward-weighted regression
        # 更完整做法：用 R 作 importance weight
        R_shifted = R - R.min() + 1e-6   # 確保非負
        w = R_shifted / R_shifted.sum()

        self._models = []
        for i in range(len(ACTION_KEYS)):
            y = A[:, i]   # observed action value
            # 加權目標：reward-weighted expected action
            y_target = np.full_like(y, float(np.sum(w * y)))
            # 用完整 y 訓練，以 reward 高的樣本拉近
            y_blend = (1 - w) * y + w * y_target   # soft target

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if _HAS_XGB:
                    m = XGBRegressor(**self.xgb_kwargs)
                    m.fit(X, y_blend)
                else:
                    # Ridge fallback: 包成 lambda
                    X_train, y_train = X.copy(), y_blend.copy()
                    m = lambda xp, _X=X_train, _y=y_train: _ridge_fit_predict(
                        _X, _y, np.atleast_2d(xp)
                    ).flatten()
            self._models.append(m)

        self._n_since_fit = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, ctx: np.ndarray) -> CalibAction:
        """給定 context，回傳校正 action。模型未訓練前回傳 zero action。"""
        if self._models is None or len(self._buffer) < self.min_train:
            action = CalibAction.zero()
        else:
            x = ctx.reshape(1, -1)
            deltas = []
            for i, m in enumerate(self._models):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if _HAS_XGB:
                        val = float(m.predict(x)[0])
                    else:
                        val = float(m(x))
                lo, hi = ACTION_CLIP[ACTION_KEYS[i]]
                deltas.append(float(np.clip(val, lo, hi)))
            action = CalibAction.from_array(np.array(deltas))

        # Gaussian exploration noise（訓練期間）
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
        """儲存一筆經驗，並在條件達成時重新訓練模型。"""
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
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._buffer          = state["buffer"]
        self._models          = state.get("models")
        self.n_experiences    = state.get("n_experiences", len(self._buffer))
        self.explore_std      = state.get("explore_std",     self.explore_std)
        self.min_train        = state.get("min_train",        self.min_train)
        self.update_interval  = state.get("update_interval", self.update_interval)
        self.xgb_kwargs       = state.get("xgb_kwargs",      self.xgb_kwargs)
        self._n_since_fit     = 0
