"""
calibrator.py  v6.3
===================
AdaptiveCalibrator：ES（Evolution Strategy）策略更新 + 連續 RL 閉璴支援。

v6.3:
  _compute_reward() kurt weight floor + _KURT_N_BASE 調整。
    問題：
      v6.2 的 weight_k = min(n_bars,120)/120 在 step=20 時為 0.167，
      calibrator 對 kurt 的修正梓度極小，14000+ 筆經驗後 kurt_err
      仍然沒有顯著下降。
    修正：
      1. _KURT_N_BASE: 120 -> 60（折扣曲線的膝點調到 1 個 lookback window）。
      2. weight_k 增加 floor 0.33：
           weight_k = max(min(n_bars, _KURT_N_BASE) / _KURT_N_BASE, 0.33)
         step=20 時：0.167 -> 0.33，kurt 模擬訊號翻倍。
         step=60 時：1.0（不受影響）。
      3. PARAM_SAFE target_ek 上限 60 -> 120：
         允許 calibrator 對 UNH 等極高 kurtosis 品種推高 target_ek，
         匹配 stat_process._TARGET_EK_MAX=60 + ek_oversample_adj 路徑。

v6.2 (B方案): kurt_err 加入樣本量折扣。
v6.1 (P1)：
  1. PARAM_SAFE["target_ek"] 上限 15.0 → 60.0。
  2. d_target_ek clip: (-0.50, 0.50) → (-0.30, 0.30)
v6 主要變更：
  1. build_context 擴充至 10 維。
  2. reward 重新平衡。
  3. CONTEXT_KEYS 同步更新。10 個 key。24. build_context 新增 ek_oversample 關鍵字參數。
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
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
    "ek_oversample_adj",
]

ACTION_KEYS: List[str] = [
    "d_ret_std", "d_hurst", "d_target_ek", "d_vol_persistence",
]

ACTION_CLIP: Dict[str, tuple] = {
    "d_ret_std":          (-0.40, 0.40),
    "d_hurst":            (-0.15, 0.15),
    "d_target_ek":        (-0.30, 0.30),
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
    "target_ek":        (1.0,   120.0),  # v6.3: raised from 60 to allow high-kurt adjustment
    "vol_persistence":  (0.0,   0.85),
}

_N_ACTIONS = len(ACTION_KEYS)

# ES 超參
_ES_EXPLORE_INIT  = 0.15
_ES_EXPLORE_FLOOR = 0.02
_ES_DECAY_HALF    = 2000
_ES_LR            = 0.10

# v6.3: kurt discount base lowered 120->60 (one lookback window)
_KURT_N_BASE = 60
# v6.3: kurt weight floor — even at step=20 calibrator gets 33% of full signal
_KURT_WEIGHT_FLOOR = 0.33


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
# ESPolicy
# ---------------------------------------------------------------------------

class ESPolicy:
    def __init__(self):
        self.mean_vec   = np.zeros(_N_ACTIONS, dtype=float)
        self._total_upd = 0

    @property
    def explore_std(self) -> float:
        decay = 0.5 ** (self._total_upd / _ES_DECAY_HALF)
        return float(max(_ES_EXPLORE_FLOOR, _ES_EXPLORE_INIT * decay))

    def propose(self) -> np.ndarray:
        raw = self.mean_vec + np.random.normal(0, self.explore_std, _N_ACTIONS)
        for i, key in enumerate(ACTION_KEYS):
            lo, hi = ACTION_CLIP[key]
            raw[i] = float(np.clip(raw[i], lo, hi))
        return raw

    def update(self, actions: np.ndarray, rewards: np.ndarray) -> None:
        if len(actions) == 0:
            return
        r_shifted = rewards - rewards.min() + 1e-8
        w = r_shifted / r_shifted.sum()
        weighted_mean = (actions * w[:, None]).sum(axis=0)
        self.mean_vec = (1 - _ES_LR) * self.mean_vec + _ES_LR * weighted_mean
        for i, key in enumerate(ACTION_KEYS):
            lo, hi = ACTION_CLIP[key]
            self.mean_vec[i] = float(np.clip(self.mean_vec[i], lo, hi))
        self._total_upd += len(actions)


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
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
# RidgeModel  (XGB fallback)
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
# AdaptiveCalibrator  v6.3
# ---------------------------------------------------------------------------

class AdaptiveCalibrator:
    """
    v6.3: kurt weight floor + _KURT_N_BASE 60 + PARAM_SAFE target_ek 120.
    v6.2: kurt_err sample-size discount.
    v6.1: PARAM_SAFE target_ek 15->60, d_target_ek clip +-0.50->+-0.30.
    v6:   build_context 10-dim + reward rebalanced.
    """

    def __init__(
        self,
        capacity:         int   = 20000,
        min_train:        int   = 50,
        update_interval:  int   = 20,
        xgb_n_estimators: int   = 80,
        xgb_max_depth:    int   = 4,
        xgb_lr:           float = 0.10,
    ):
        self.min_train       = min_train
        self.update_interval = update_interval
        self.xgb_kwargs      = dict(
            n_estimators     = xgb_n_estimators,
            max_depth        = xgb_max_depth,
            learning_rate    = xgb_lr,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            verbosity        = 0,
        )
        self._buffer         = ReplayBuffer(capacity)
        self._models: Optional[List[Any]] = None
        self._es             = ESPolicy()
        self._n_since_fit: int = 0
        self.n_experiences:  int = 0
        self._reward_history: List[float] = []

    @property
    def explore_std(self) -> float:
        return self._es.explore_std

    @staticmethod
    def build_context(
        params: Dict[str, Any],
        ek_oversample: float = 1.0,
    ) -> np.ndarray:
        return np.array([
            float(params["ret_std"]),
            float(params["ret_skew_a"]),
            float(params["ret_df"]),
            float(params["hurst_target"]),
            float(params["wick_lambda"]),
            float(params["jump_freq"]),
            float(params["vol_persistence"]),
            float(params["acf_lag1"]),
            float(np.log1p(abs(params["target_ek"]))),
            float(np.clip(ek_oversample, 1.0, 10.0)),
        ], dtype=float)

    @staticmethod
    def _compute_reward(
        std_err_pct: float,
        kurt_err:    float,
        hurst_err:   float,
        dir_hit:     float,
        n_bars:      int = _KURT_N_BASE,
    ) -> float:
        """
        v6.3: kurt weight_k has a floor of _KURT_WEIGHT_FLOOR (0.33).
          weight_k = max(min(n_bars, _KURT_N_BASE) / _KURT_N_BASE, _KURT_WEIGHT_FLOOR)
          At step=20: was 20/120=0.167, now max(20/60, 0.33) = max(0.33, 0.33) = 0.33.
          At step=60: max(60/60, 0.33) = 1.0 (unchanged).
          At step=20 with old _KURT_N_BASE=120: was 0.167, now 0.33 — doubles signal.
        """
        weight_k = float(max(
            min(n_bars, _KURT_N_BASE) / _KURT_N_BASE,
            _KURT_WEIGHT_FLOOR,
        ))
        r = -(
            0.05 * float(std_err_pct)
            + 0.70 * weight_k * float(np.log1p(kurt_err)) / 3.0
            + 0.15 * float(hurst_err) / 0.05
            - 0.10 * float(dir_hit)
        )
        return float(np.clip(r, -10.0, 2.0))

    def _fit_models(self) -> None:
        if len(self._buffer) < self.min_train:
            return
        X = self._buffer.contexts
        A = self._buffer.actions
        R = self._buffer.rewards
        R_shifted = R - R.min() + 1e-6
        w = R_shifted / R_shifted.sum()

        self._es.update(A, R)

        if not _HAS_XGB or len(self._buffer) < self.min_train * 2:
            models = []
            for j in range(_N_ACTIONS):
                m = RidgeModel(alpha=1.0)
                m.fit(X, A[:, j])
                models.append(m)
            self._models = models
            return

        models = []
        for j in range(_N_ACTIONS):
            m = XGBRegressor(**self.xgb_kwargs)
            m.fit(X, A[:, j], sample_weight=w)
            models.append(m)
        self._models = models

    def predict(self, ctx: np.ndarray) -> "CalibAction":
        arr = self._es.propose()
        if self._models is not None:
            ctx2 = np.atleast_2d(ctx)
            for j, model in enumerate(self._models):
                try:
                    pred = float(np.atleast_1d(model.predict(ctx2))[0])
                    lo, hi = ACTION_CLIP[ACTION_KEYS[j]]
                    arr[j] = float(np.clip(pred + self._es.explore_std * np.random.randn(), lo, hi))
                except Exception:
                    pass
        return CalibAction.from_array(arr)

    def record(
        self,
        context:     np.ndarray,
        action:      "CalibAction",
        std_err_pct: float,
        kurt_err:    float,
        hurst_err:   float,
        dir_hit:     float,
        n_bars:      int = _KURT_N_BASE,
    ) -> None:
        """v6.2+: n_bars forwarded to _compute_reward for kurt discount."""
        reward = self._compute_reward(std_err_pct, kurt_err, hurst_err, dir_hit, n_bars=n_bars)
        self._buffer.push(context, action.to_array(), reward)
        self._reward_history.append(reward)
        self.n_experiences += 1
        self._n_since_fit  += 1
        if self._n_since_fit >= self.update_interval:
            self._fit_models()
            self._n_since_fit = 0

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "AdaptiveCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self) -> Dict[str, Any]:
        rh = np.array(self._reward_history)
        return {
            "n_experiences": self.n_experiences,
            "buffer_size":   len(self._buffer),
            "explore_std":   round(self.explore_std, 4),
            "mean_reward":   round(float(rh.mean()), 4) if len(rh) > 0 else None,
            "recent_reward": round(float(rh[-20:].mean()), 4) if len(rh) >= 20 else None,
            "es_mean_vec":   [round(float(v), 4) for v in self._es.mean_vec],
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"AdaptiveCalibrator(n={s['n_experiences']}"
            f"  buf={s['buffer_size']}"
            f"  explore_std={s['explore_std']:.4f})"
            f"  mean_reward={s['mean_reward']}"
            f"  recent={s['recent_reward']}"
            f"  es_mean={s['es_mean_vec']})"
        )
