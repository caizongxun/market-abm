"""
calibrator.py  v5
=================
AdaptiveCalibrator：ES（Evolution Strategy）策略更新 + 連續 RL 閉環支援。

v5 主要變更：
  1. ES policy：每個 action dimension 維護 (mean, std) 對，
     propose() 從學習到的分佈採樣，不再是純 random noise。
  2. update_es()：reward-weighted mean update（CMA-ES 簡化版），
     每個 window 結束後立即更新 policy mean。
  3. explore_std 動態 decay：0.15 → floor 0.02，
     experience 越多探索越收斂。
  4. composite reward 重新平衡：
     std_err 權重降低（hard renorm 已保證），kurt_err 升至主導。
  5. ReplayBuffer capacity 5000 → 20000。
  6. 新增 summary() 方便 train_rl.py 打印進度。

v4 patch（保留）：
  kurt_err log1p 壓縮；build_context log1p(target_ek)。
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
]

ACTION_KEYS: List[str] = [
    "d_ret_std", "d_hurst", "d_target_ek", "d_vol_persistence",
]

# action 代表對參數的相對調整（乘數 delta），clip 為安全邊界
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
_ES_EXPLORE_INIT  = 0.15    # 初始探索 std
_ES_EXPLORE_FLOOR = 0.02    # 最低探索 std
_ES_DECAY_HALF    = 2000    # 每 2000 筆經驗 std 減半
_ES_LR            = 0.10    # mean update learning rate


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
# ESPolicy  —  per-context Gaussian policy，reward-weighted mean update
# ---------------------------------------------------------------------------

class ESPolicy:
    """
    簡化 CMA-ES：維護每個 action dim 的 (mean, explore_std)。
    - propose(ctx)：從 N(mean, std^2) 採樣 action
    - update(actions, rewards)：reward-weighted mean update
    - explore_std 隨 total_updates 指數衰減
    """

    def __init__(self):
        self.mean_vec   = np.zeros(_N_ACTIONS, dtype=float)
        self._total_upd = 0

    @property
    def explore_std(self) -> float:
        decay = 0.5 ** (self._total_upd / _ES_DECAY_HALF)
        return float(max(_ES_EXPLORE_FLOOR,
                         _ES_EXPLORE_INIT * decay))

    def propose(self) -> np.ndarray:
        """從當前 policy 分佈採樣一個 action vector。"""
        raw = self.mean_vec + np.random.normal(0, self.explore_std, _N_ACTIONS)
        for i, key in enumerate(ACTION_KEYS):
            lo, hi = ACTION_CLIP[key]
            raw[i] = float(np.clip(raw[i], lo, hi))
        return raw

    def update(self, actions: np.ndarray, rewards: np.ndarray) -> None:
        """
        actions: (N, n_actions)
        rewards: (N,)
        reward-weighted mean update：往高 reward action 的均值方向走。
        """
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
# AdaptiveCalibrator  v5
# ---------------------------------------------------------------------------

class AdaptiveCalibrator:
    """
    v5: ES policy update + 連續 RL 閉環。

    主要流程：
      1. predict(ctx)  → propose action from ES policy
      2. apply action  → generate() with adjusted params
      3. record(...)   → push to buffer, update ES policy mean
      4. _fit_models() → 每 update_interval 筆用 XGB/Ridge 學習 context→action mapping
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
        # rolling reward stats for live monitoring
        self._reward_history: List[float] = []

    @property
    def explore_std(self) -> float:
        return self._es.explore_std

    @staticmethod
    def build_context(params: Dict[str, Any]) -> np.ndarray:
        """9-dim context vector。target_ek 用 log1p 壓縮（v4 Patch-3 保留）。"""
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
        ], dtype=float)

    @staticmethod
    def _compute_reward(
        std_err_pct: float,
        kurt_err:    float,
        hurst_err:   float,
        dir_hit:     float,
    ) -> float:
        """
        v5 reward 重新平衡：
          std_err 權重降低（hard renorm 已保證精確），
          kurt_err 升至主導（目前 20x 超標是主戰場），
          dir_hit 提高權重（策略訓練的最終目標）。

          r = -(0.10*std_err + 0.50*log1p(kurt)/3 + 0.20*hurst/0.05 - 0.20*dir_hit)
        """
        r = -(
            0.10 * float(std_err_pct)
            + 0.50 * float(np.log1p(kurt_err)) / 3.0
            + 0.20 * float(hurst_err) / 0.05
            - 0.20 * float(dir_hit)
        )
        return float(np.clip(r, -10.0, 2.0))

    def _fit_models(self) -> None:
        """用 buffer 裡的 (context, action, reward) 訓練 action predictor。"""
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
        """
        v5: 先用 XGB/Ridge 給出 base action（如果已訓練），
        再疊加 ES policy noise（從學習到的分佈採樣）。
        """
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

        # ES noise：從 policy mean + explore_std 採樣
        es_sample = self._es.propose()
        # blend: base (XGB) + ES perturbation
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

        # ES policy mean update（每筆立即更新，小 batch=1）
        self._es.update(
            actions = act_arr.reshape(1, -1),
            rewards = np.array([reward]),
        )

        # XGB/Ridge 定期重新 fit
        if (self._n_since_fit >= self.update_interval
                and len(self._buffer) >= self.min_train):
            self._fit_models()

    def summary(self) -> Dict[str, Any]:
        """回傳當前訓練狀態，供 train_rl.py 打印。"""
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
        if self._models is None and len(self._buffer) >= self.min_train:
            self._fit_models()
        print(f"[calib] loaded  n_exp={self.n_experiences}  buf={len(self._buffer)}  "
              f"explore_std={self.explore_std:.4f}")
