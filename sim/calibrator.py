"""
calibrator.py  v6
=================
AdaptiveCalibrator：ES（Evolution Strategy）策略更新 + 連續 RL 閉環支援。

v6 主要變更：
  1. build_context 擴充至 10 維：新增 ek_oversample_adj。
     calibrator 現在能看到當前 window 的自適應超取樣倍率，
     從而學到「ek_adj 偏低 → 應該提高 d_target_ek」這個映射。
  2. reward 重新平衡（kurtosis 升至主導）：
     Old: 0.10*std + 0.50*log1p(kurt)/3 + 0.20*hurst - 0.20*dir
     New: 0.05*std + 0.70*log1p(kurt)/3 + 0.15*hurst - 0.10*dir
     在 kurt_err=6.65 時 penalty 從 0.67 提高到 0.93。
  3. CONTEXT_KEYS 同步更新（10 個 key）。
  4. build_context 新增 ek_oversample 關鍵字參數（預設 1.0 以向後相容）。

v5（保留）：
  ES policy update + 連續 RL 閉環。
  explore_std 動態 decay 0.15→0.02。
  ReplayBuffer capacity 20000。
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
    "ek_oversample_adj",   # v6: 第 10 維，adaptive 超取樣倍率
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
    "target_ek":        (1.0,   15.0),
    "vol_persistence":  (0.0,   0.85),
}

_N_ACTIONS = len(ACTION_KEYS)

# ES 超參
_ES_EXPLORE_INIT  = 0.15
_ES_EXPLORE_FLOOR = 0.02
_ES_DECAY_HALF    = 2000
_ES_LR            = 0.10


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
# AdaptiveCalibrator  v6
# ---------------------------------------------------------------------------

class AdaptiveCalibrator:
    """
    v6: build_context 10-dim (+ ek_oversample_adj) + reward 重新平衡。

    主要流程：
      1. predict(ctx)  → propose action from ES policy
      2. apply action  → generate() with adjusted params
      3. record(...)   → push to buffer, update ES policy mean
      4. _fit_models() → 每 update_interval 筆用 XGB/Ridge 學習
      5. save/load     → 持久化，支援跨 session warm-start
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
        """
        v6: 10-dim context vector。
        第 10 維新增 ek_oversample_adj，讓模型能觀察到當前的
        adaptive 超取樣倍率，從而學到修正 d_target_ek 的方向。
        ek_oversample 預設 1.0 以向後相容舊的呼叫端。
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
            float(np.log1p(abs(params["target_ek"]))),
            float(np.clip(ek_oversample, 1.0, 8.0)),   # v6: 第 10 維
        ], dtype=float)

    @staticmethod
    def _compute_reward(
        std_err_pct: float,
        kurt_err:    float,
        hurst_err:   float,
        dir_hit:     float,
    ) -> float:
        """
        v6 reward 重新平衡：kurtosis 升至主導信號。

          Old: -(0.10*std + 0.50*log1p(kurt)/3 + 0.20*hurst/0.05 - 0.20*dir)
          New: -(0.05*std + 0.70*log1p(kurt)/3 + 0.15*hurst/0.05 - 0.10*dir)

        在 kurt_err=6.65 時：
          Old penalty = 0.50 * log1p(6.65)/3 ≈ 0.67
          New penalty = 0.70 * log1p(6.65)/3 ≈ 0.93

        std_err 權重從 0.10 降至 0.05（hard renorm 已保證精確）。
        dir_hit 從 0.20 降至 0.10（本輪主戰場是 kurtosis，dir 次要）。
        """
        r = -(
            0.05 * float(std_err_pct)
            + 0.70 * float(np.log1p(kurt_err)) / 3.0
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
        self._models = []
        for i in range(_N_ACTIONS):
            y        = A[:, i]
            y_target = float(np.sum(w * y))
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
        if self._models is not None and len(self._buffer) >= self.min_train:
            x = ctx.reshape(1, -1)
            base = []
            for i, m in enumerate(self._models):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val = float(m.predict(x).flat[0])
                lo, hi = ACTION_CLIP[ACTION_KEYS[i]]
                base.append(float(np.clip(val, lo, hi)))
            base_arr = np.array(base)
        else:
            base_arr = np.zeros(_N_ACTIONS)

        es_sample = self._es.propose()
        blend_w = float(np.clip(len(self._buffer) / max(self.min_train * 4, 1), 0.0, 0.8))
        a_arr = (1 - blend_w) * es_sample + blend_w * base_arr
        for i, key in enumerate(ACTION_KEYS):
            lo, hi = ACTION_CLIP[key]
            a_arr[i] = float(np.clip(a_arr[i], lo, hi))
        return CalibAction.from_array(a_arr)

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
        act_arr = action.to_array()
        self._buffer.push(context, act_arr, reward)
        self._reward_history.append(reward)
        if len(self._reward_history) > 500:
            self._reward_history = self._reward_history[-500:]
        self.n_experiences  += 1
        self._n_since_fit   += 1

        self._es.update(
            actions = act_arr.reshape(1, -1),
            rewards = np.array([reward]),
        )

        if (self._n_since_fit >= self.update_interval
                and len(self._buffer) >= self.min_train):
            self._fit_models()

    def summary(self) -> Dict[str, Any]:
        h = self._reward_history
        return {
            "n_exp":       self.n_experiences,
            "buf_size":    len(self._buffer),
            "explore_std": round(self.explore_std, 4),
            "es_mean":     self._es.mean_vec.round(4).tolist(),
            "reward_last":  round(h[-1], 4)   if h else None,
            "reward_50":   round(float(np.mean(h[-50:])),  4) if len(h) >= 50  else None,
            "reward_200":  round(float(np.mean(h[-200:])), 4) if len(h) >= 200 else None,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "buffer":          self._buffer,
                "models":          self._models,
                "es":              self._es,
                "n_experiences":   self.n_experiences,
                "min_train":       self.min_train,
                "update_interval": self.update_interval,
                "xgb_kwargs":      self.xgb_kwargs,
                "reward_history":  self._reward_history,
            }, f)

    def load(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, Exception) as exc:
            print(f"[calib] WARNING: corrupt pkl ({exc}), starting fresh.")
            try:
                os.remove(path)
            except OSError:
                pass
            return
        self._buffer          = state["buffer"]
        self._models          = state.get("models")
        self._es              = state.get("es", ESPolicy())
        self.n_experiences    = state.get("n_experiences", len(self._buffer))
        self.min_train        = state.get("min_train",        self.min_train)
        self.update_interval  = state.get("update_interval", self.update_interval)
        self.xgb_kwargs       = state.get("xgb_kwargs",      self.xgb_kwargs)
        self._reward_history  = state.get("reward_history",  [])
        self._n_since_fit     = 0

        # v6 backward compat: 舊的 pkl 存的是 9-dim context，
        # 如果 buffer 非空且 dim=9，清空 buffer 強制重新累積 10-dim 資料
        if len(self._buffer) > 0:
            sample_ctx = self._buffer._ctx[0]
            if np.asarray(sample_ctx).shape[0] != 10:
                print(f"[calib] context dim mismatch (got {np.asarray(sample_ctx).shape[0]}, need 10). "
                      f"Clearing buffer and models to avoid shape error.")
                self._buffer  = ReplayBuffer(self._buffer.capacity)
                self._models  = None
                self.n_experiences = 0
                self._reward_history = []
                self._n_since_fit = 0
                return

        if self._models is None and len(self._buffer) >= self.min_train:
            self._fit_models()
        print(f"[calib] loaded  n_exp={self.n_experiences}  buf={len(self._buffer)}  "
              f"explore_std={self.explore_std:.4f}")
