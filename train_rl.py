"""
train_rl.py
===========
持續 RL 訓練入口：不斷從歷史資料取 rolling window，
讓 AdaptiveCalibrator 的 ES policy 越跑越準。

用法：
  python train_rl.py                          # 預設 500 episodes
  python train_rl.py --episodes 2000 --lookback 60 --step 20
  python train_rl.py --episodes 200  --no-calib-load   # 從頭訓練

流程（每 episode）：
  1. 隨機選一個 symbol 和一個 start date
  2. fetch / 讀取 parquet cache
  3. rolling_fit_generate()：每個 window 做 fit → predict → generate → record
  4. 每 save_every episodes 儲存 calibrator.pkl checkpoint
  5. 打印 live reward / rolling-50 mean
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

# 確保 project root 在 path
sys.path.insert(0, os.path.dirname(__file__))

from sim.calibrator import AdaptiveCalibrator
from sim.stat_process import rolling_fit_generate


# ---------------------------------------------------------------------------
# 訓練用品種池
# ---------------------------------------------------------------------------

TRAIN_SYMBOLS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
    "JPM", "BAC", "GS", "MS", "WFC",
    "SPY", "QQQ", "IWM",
    "XOM", "CVX", "GLD", "TLT",
    "AVGO", "CAT", "GE", "JNJ", "PFE", "UNH",
]

DATA_START = "2018-01-01"
DATA_END   = datetime.today().strftime("%Y-%m-%d")

DEFAULT_CALIB_PATH = "models/calibrator.pkl"


# ---------------------------------------------------------------------------
# fetch helper（重用 run_sim.py 邏輯，避免重複下載）
# ---------------------------------------------------------------------------

def _fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    """讀 parquet cache；miss 時用 yfinance 下載並存檔。"""
    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, f"{symbol}_{start}_{end}.parquet")
    if os.path.exists(fname):
        df = pd.read_parquet(fname)
        return df
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed: pip install yfinance")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(symbol, start=start, end=end,
                         auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"yfinance returned empty data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna().reset_index(drop=True)
    df.to_parquet(fname)
    return df


# ---------------------------------------------------------------------------
# 單 episode
# ---------------------------------------------------------------------------

def run_episode(
    symbol:      str,
    df_full:     pd.DataFrame,
    calibrator:  AdaptiveCalibrator,
    lookback:    int,
    step:        int,
    min_bars:    int,
    rng:         np.random.Generator,
) -> dict:
    """
    從 df_full 中隨機取一段時間窗口，跑 rolling_fit_generate，
    回傳 episode summary。
    """
    n = len(df_full)
    min_len = lookback + step * 3   # 至少要有 3 個 forward window
    if n < min_len:
        return {"skip": True, "reason": "not enough bars"}

    max_start = n - min_len
    start_idx = int(rng.integers(0, max_start + 1))
    df_slice  = df_full.iloc[start_idx:].reset_index(drop=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, param_log = rolling_fit_generate(
            df_real    = df_slice,
            lookback   = lookback,
            step       = step,
            seed       = int(rng.integers(0, 2**31)),
            verbose    = False,
            use_adapt  = False,        # OnlineRidgePredictor 關閉（ES 接管）
            calibrator = calibrator,
        )

    # 從 param_log 取最後一筆 summary
    summary_rows = [r for r in param_log if r.get("_summary")]
    window_rows  = [r for r in param_log if not r.get("_summary")]
    n_windows    = len(window_rows)
    if n_windows == 0:
        return {"skip": True, "reason": "no windows"}

    return {
        "skip":       False,
        "symbol":     symbol,
        "n_windows":  n_windows,
        "start_idx":  start_idx,
    }


# ---------------------------------------------------------------------------
# 主訓練 loop
# ---------------------------------------------------------------------------

def train(
    episodes:       int   = 500,
    lookback:       int   = 60,
    step:           int   = 20,
    save_every:     int   = 50,
    calib_path:     str   = DEFAULT_CALIB_PATH,
    load_calib:     bool  = True,
    seed:           int   = 0,
) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # 建立 calibrator，選擇性 warm-start
    calibrator = AdaptiveCalibrator()
    if load_calib and os.path.exists(calib_path):
        calibrator.load(calib_path)
    else:
        print(f"[train] starting fresh calibrator (path={calib_path})")

    # 預載所有品種資料（避免每 episode 重複 IO）
    print(f"[train] pre-fetching {len(TRAIN_SYMBOLS)} symbols ...")
    symbol_data = {}
    for sym in TRAIN_SYMBOLS:
        try:
            df = _fetch(sym, DATA_START, DATA_END)
            if len(df) >= lookback + step * 3:
                symbol_data[sym] = df
                print(f"  {sym:6s}  {len(df)} bars")
        except Exception as e:
            print(f"  {sym:6s}  SKIP ({e})")
    available = list(symbol_data.keys())
    if not available:
        print("[train] ERROR: no symbols available")
        return
    print(f"[train] ready: {len(available)} symbols, starting {episodes} episodes\n")

    ok_count   = 0
    skip_count = 0

    for ep in range(1, episodes + 1):
        sym    = random.choice(available)
        df_sym = symbol_data[sym]

        result = run_episode(
            symbol     = sym,
            df_full    = df_sym,
            calibrator = calibrator,
            lookback   = lookback,
            step       = step,
            min_bars   = lookback + step * 3,
            rng        = rng,
        )

        if result.get("skip"):
            skip_count += 1
            continue
        ok_count += 1

        # live print
        summ = calibrator.summary()
        r_last = summ["reward_last"]
        r50    = summ["reward_50"]
        r200   = summ["reward_200"]
        r50_str  = f"{r50:+.4f}"  if r50  is not None else "   n/a"
        r200_str = f"{r200:+.4f}" if r200 is not None else "   n/a"
        print(
            f"[ep {ep:5d}/{episodes}]  {sym:5s}  "
            f"n_exp={summ['n_exp']:6d}  explore={summ['explore_std']:.4f}  "
            f"r_last={r_last:+.4f}  r50={r50_str}  r200={r200_str}  "
            f"es_mean={summ['es_mean']}"
        )

        # checkpoint
        if ep % save_every == 0:
            calibrator.save(calib_path)
            print(f"  -> checkpoint saved: {calib_path}  "
                  f"(n_exp={calibrator.n_experiences})")

    # 最終儲存
    calibrator.save(calib_path)
    print(f"\n[train] done  ok={ok_count}  skip={skip_count}")
    print(f"[train] final calibrator: n_exp={calibrator.n_experiences}  "
          f"explore_std={calibrator.explore_std:.4f}")
    final_summ = calibrator.summary()
    print(f"[train] ES policy mean: {final_summ['es_mean']}")
    print(f"[train] reward_200 avg: {final_summ['reward_200']}")
    print(f"[train] saved -> {calib_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL training loop for AdaptiveCalibrator")
    parser.add_argument("--episodes",      type=int,   default=500)
    parser.add_argument("--lookback",      type=int,   default=60)
    parser.add_argument("--step",          type=int,   default=20)
    parser.add_argument("--save-every",    type=int,   default=50)
    parser.add_argument("--calib-path",    type=str,   default=DEFAULT_CALIB_PATH)
    parser.add_argument("--no-calib-load", action="store_true",
                        help="ignore existing pkl and train from scratch")
    parser.add_argument("--seed",          type=int,   default=0)
    args = parser.parse_args()

    train(
        episodes   = args.episodes,
        lookback   = args.lookback,
        step       = args.step,
        save_every = args.save_every,
        calib_path = args.calib_path,
        load_calib = not args.no_calib_load,
        seed       = args.seed,
    )
