"""
calibrate.py
============
ParamCalibrator: 對歷史資料做網格搜尋，找出最佳
  impact_coeff / momentum_scale / decay

目標函數
--------
  loss = w_hurst * |sim_hurst - real_hurst|
       + w_dir   * (1 - direction_hit_rate)
       + w_kurt  * |sim_kurtosis - real_kurtosis| / (|real_kurtosis| + 1)

較低 loss = 更接近真實統計特性。

使用方式
--------
  python -m sim.calibrate --symbol AAPL --start 2024-01-01 --end 2025-06-01

  或在程式碼中：
    from sim.calibrate import ParamCalibrator
    cal = ParamCalibrator(df_train, df_real_future)
    best = cal.run(n_sims=30)
    print(best)
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.simulation import run_simulation
from sim.metrics import compare, hurst_exponent, log_returns


# ---------------------------------------------------------------------------
# Default search grid
# ---------------------------------------------------------------------------
DEFAULT_GRID: dict[str, list] = {
    "impact_coeff":    [0.0008, 0.0015, 0.0025, 0.004],
    "momentum_scale":  [0.5, 1.0, 2.0, 3.0],
    "decay":           [0.90, 0.93, 0.97, 0.99],
}


# ---------------------------------------------------------------------------
class ParamCalibrator:
    """
    Parameters
    ----------
    df_train : pd.DataFrame
        Historical OHLCV used as agent warmup context.
    df_real_future : pd.DataFrame
        Real future bars to compare against (same length as sim_bars).
    sim_bars : int
        Number of bars to simulate per evaluation.
    warmup_bars : int
        Warmup bars passed to run_simulation.
    w_hurst : float
        Weight on |sim_hurst - real_hurst| in loss.
    w_dir : float
        Weight on (1 - direction_hit_rate) in loss.
    w_kurt : float
        Weight on normalised kurtosis error in loss.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_real_future: pd.DataFrame,
        sim_bars: int = 60,
        warmup_bars: int = 100,
        w_hurst: float = 1.0,
        w_dir:   float = 1.0,
        w_kurt:  float = 0.5,
        n_institution: int = 5,
        n_momentum: int = 40,
        n_random: int = 100,
        n_contrarian: int = 15,
        momentum_window_fast: int = 5,
        momentum_window_slow: int = 20,
        path_floor_pct: float = 0.30,
        intra_noise_scale: float = 1.0,
    ):
        self.df_train        = df_train
        self.df_real_future  = df_real_future
        self.sim_bars        = sim_bars
        self.warmup_bars     = warmup_bars
        self.w_hurst         = w_hurst
        self.w_dir           = w_dir
        self.w_kurt          = w_kurt
        self.n_institution   = n_institution
        self.n_momentum      = n_momentum
        self.n_random        = n_random
        self.n_contrarian    = n_contrarian
        self.momentum_window_fast = momentum_window_fast
        self.momentum_window_slow = momentum_window_slow
        self.path_floor_pct  = path_floor_pct
        self.intra_noise_scale = intra_noise_scale

        # Pre-compute real stats once
        real_rets = log_returns(df_real_future["Close"].values)
        self.real_hurst = hurst_exponent(real_rets)
        from scipy import stats as scipy_stats
        self.real_kurtosis = float(scipy_stats.kurtosis(real_rets))

    def _loss(self, metrics: dict) -> float:
        sim  = metrics["sim"]
        loss = (
            self.w_hurst * abs(sim["hurst"] - self.real_hurst)
            + self.w_dir * (1.0 - metrics["direction_hit_rate"])
            + self.w_kurt * abs(sim["kurtosis"] - self.real_kurtosis)
              / (abs(self.real_kurtosis) + 1.0)
        )
        return float(loss)

    def evaluate(
        self,
        impact_coeff: float,
        momentum_scale: float,
        decay: float,
        n_sims: int = 20,
        seed: int = 0,
    ) -> dict[str, Any]:
        """
        Run n_sims paths with given params, average metrics over paths,
        return loss + detailed stats.
        """
        paths = []
        for i in range(n_sims):
            _, df_sim = run_simulation(
                df_real=self.df_train,
                sim_bars=self.sim_bars,
                warmup_bars=self.warmup_bars,
                impact_coeff=impact_coeff,
                intra_noise_scale=self.intra_noise_scale,
                momentum_scale=momentum_scale,
                bias_decay=decay,
                use_momentum_init=True,
                auto_drift=True,
                n_institution=self.n_institution,
                n_momentum=self.n_momentum,
                n_random=self.n_random,
                n_contrarian=self.n_contrarian,
                path_floor_pct=self.path_floor_pct,
                momentum_window_fast=self.momentum_window_fast,
                momentum_window_slow=self.momentum_window_slow,
                seed=seed + i,
            )
            paths.append(df_sim)

        # Median path for deterministic metrics
        close_mat = np.vstack([p["Close"].values for p in paths])
        med_close = np.median(close_mat, axis=0)
        df_med = paths[0].copy()
        df_med["Close"] = med_close

        m = compare(self.df_real_future, df_med, print_report=False)
        loss = self._loss(m)

        # P(up) across paths
        final_rets = (close_mat[:, -1] - close_mat[:, 0]) / close_mat[:, 0]
        p_up = float((final_rets > 0).mean())

        return {
            "impact_coeff":    impact_coeff,
            "momentum_scale":  momentum_scale,
            "decay":           decay,
            "loss":            loss,
            "hurst_sim":       m["sim"]["hurst"],
            "hurst_real":      self.real_hurst,
            "kurtosis_sim":    m["sim"]["kurtosis"],
            "kurtosis_real":   self.real_kurtosis,
            "direction_hit":   m["direction_hit_rate"],
            "p_up":            p_up,
        }

    def run(
        self,
        grid: dict[str, list] | None = None,
        n_sims: int = 20,
        seed: int = 0,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Grid search over params. Returns best param dict by loss.
        """
        if grid is None:
            grid = DEFAULT_GRID

        keys   = list(grid.keys())
        combos = list(itertools.product(*[grid[k] for k in keys]))
        total  = len(combos)

        if verbose:
            print(f"[calibrate] grid: {dict(zip(keys, [len(grid[k]) for k in keys]))}")
            print(f"[calibrate] total combos: {total}  x  {n_sims} sims each = {total * n_sims} runs")
            print(f"[calibrate] real_hurst={self.real_hurst:.4f}  real_kurtosis={self.real_kurtosis:.4f}")
            print()

        results = []
        for idx, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            res = self.evaluate(**params, n_sims=n_sims, seed=seed)
            results.append(res)
            if verbose:
                print(
                    f"  [{idx+1:>3}/{total}]  "
                    f"impact={params['impact_coeff']:.4f}  "
                    f"scale={params['momentum_scale']:.1f}  "
                    f"decay={params['decay']:.2f}  "
                    f"loss={res['loss']:.4f}  "
                    f"hurst={res['hurst_sim']:.3f}  "
                    f"dir_hit={res['direction_hit']:.3f}  "
                    f"p_up={res['p_up']:.2f}"
                )

        results.sort(key=lambda x: x["loss"])
        best = results[0]

        if verbose:
            print()
            print("[calibrate] === BEST ===")
            print(f"  impact_coeff   = {best['impact_coeff']}")
            print(f"  momentum_scale = {best['momentum_scale']}")
            print(f"  decay          = {best['decay']}")
            print(f"  loss           = {best['loss']:.4f}")
            print(f"  hurst   sim/real = {best['hurst_sim']:.4f} / {best['hurst_real']:.4f}")
            print(f"  kurtosis sim/real = {best['kurtosis_sim']:.4f} / {best['kurtosis_real']:.4f}")
            print(f"  direction_hit  = {best['direction_hit']:.4f}")
            print(f"  p_up           = {best['p_up']:.2%}")

        self.results_ = results
        self.best_    = best
        return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    p = argparse.ArgumentParser(description="ParamCalibrator CLI")
    p.add_argument("--symbol",  default="AAPL")
    p.add_argument("--start",   default="2024-01-01")
    p.add_argument("--end",     default=None)
    p.add_argument("--bars",    type=int, default=60)
    p.add_argument("--warmup",  type=int, default=100)
    p.add_argument("--n-sims",  type=int, default=20,
                   help="Paths per combo (more = stabler metrics, slower)")
    p.add_argument("--n-rand",  type=int, default=100)
    p.add_argument("--n-cont",  type=int, default=15)
    p.add_argument("--seed",    type=int, default=0)
    args = p.parse_args()

    from data.fetch import fetch_ohlcv
    from datetime import date

    end_date = args.end or date.today().strftime("%Y-%m-%d")
    print(f"[fetch] {args.symbol}  {args.start} ~ {end_date}")
    df_all = fetch_ohlcv(args.symbol, args.start, end_date)

    if len(df_all) < args.warmup + args.bars:
        print(f"[error] Not enough data ({len(df_all)} rows).")
        sys.exit(1)

    df_train       = df_all.iloc[:-(args.bars)].copy()
    df_real_future = df_all.iloc[-(args.bars):].copy().reset_index(drop=True)
    print(f"[data] train={len(df_train)}  future={len(df_real_future)}")

    cal  = ParamCalibrator(
        df_train, df_real_future,
        sim_bars=args.bars,
        warmup_bars=args.warmup,
        n_random=args.n_rand,
        n_contrarian=args.n_cont,
    )
    best = cal.run(n_sims=args.n_sims, seed=args.seed)

    # Print run_sim.py equivalent command
    print()
    print("[calibrate] === Equivalent run_sim.py command ===")
    print(
        f"python run_sim.py --symbol {args.symbol} "
        f"--start {args.start} --end {end_date} "
        f"--bars {args.bars} --momentum-init --n-sims 200 --plot "
        f"--impact {best['impact_coeff']} "
        f"--momentum-scale {best['momentum_scale']} "
        f"--decay {best['decay']}"
    )


if __name__ == "__main__":
    _cli()
