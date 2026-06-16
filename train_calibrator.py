"""
train_calibrator.py
===================
用大量隨機 trial 離線預訓練 AdaptiveCalibrator，
儲存到 models/calibrator.pkl 供 bench_generalize.py 載入使用。

用法
----
  python train_calibrator.py                      # 預設 300 trials
  python train_calibrator.py --runs 500 --symbols AAPL MSFT GOOGL
  python train_calibrator.py --runs 1000 --start 2018-01-01

說明
----
  - 每 trial 結束後 calibrator 自動更新（update_interval=10）
  - 最終 save 到 --out 路徑
  - 可以多次呼叫累積訓練（每次 load 上次的結果繼續訓練）
  - 建議訓練量：300 trials ~ 15,000 筆經驗，XGBoost 效果開始顯現
    1000 trials ~ 50,000 筆，泛化能力穩定
"""
from __future__ import annotations

import argparse
import os
import random
import warnings

import numpy as np

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "JPM", "GS", "BAC", "WFC", "MS",
    "WMT", "COST", "UNH", "JNJ", "PFE",
    "XOM", "CVX", "CAT", "BA", "GE",
    "SPY", "QQQ", "IWM", "GLD", "TLT", "XLF", "XLE",
]

START_POOL = [
    "2018-01-01", "2018-07-01",
    "2019-01-01", "2019-07-01",
    "2020-01-01", "2020-07-01",
    "2021-01-01", "2021-07-01",
    "2022-01-01", "2022-07-01",
    "2023-01-01",
]


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain AdaptiveCalibrator")
    p.add_argument("--runs",     type=int,  default=300,  help="訓練 trial 數")
    p.add_argument("--lookback", type=int,  default=60)
    p.add_argument("--step",     type=int,  default=20)
    p.add_argument("--seed",     type=int,  default=1)
    p.add_argument("--out",      type=str,  default="models/calibrator.pkl")
    p.add_argument("--symbols",  type=str,  nargs="*", default=None)
    p.add_argument("--no-xgb",   action="store_true", help="強制使用 Ridge fallback")
    return p.parse_args()


def main():
    args    = parse_args()
    rng     = random.Random(args.seed)
    np_rng  = np.random.default_rng(args.seed)
    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS

    os.makedirs("models", exist_ok=True)

    from sim.calibrator import AdaptiveCalibrator
    cal = AdaptiveCalibrator(
        min_train       = 50,
        update_interval = 10,   # 訓練時更頻繁更新
        explore_std     = 0.05,
        use_xgb         = not args.no_xgb,
    )

    # 繼續累積：若已有預訓練檔案就載入
    if os.path.exists(args.out):
        cal.load(args.out)
        print(f"[train] resumed from {args.out}  ({cal.n_experiences} experiences)")
    else:
        print(f"[train] starting fresh -> {args.out}")

    from data.fetch import get_ohlcv
    from sim.stat_process import rolling_fit_generate

    total = args.runs
    ok_count = 0

    for i in range(total):
        sym   = rng.choice(symbols)
        start = rng.choice(START_POOL)
        seed  = int(np_rng.integers(0, 2**31))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_real = get_ohlcv(sym, start=start)

            if df_real is None or len(df_real) < args.lookback + args.step * 2:
                print(f"  [{i+1:4d}/{total}] skip  {sym:<6} {start}  (insufficient bars)")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rolling_fit_generate(
                    df_real    = df_real,
                    lookback   = args.lookback,
                    step       = args.step,
                    seed       = seed,
                    verbose    = False,
                    calibrator = cal,
                )

            ok_count += 1
            trained_str = "trained" if cal.is_trained else "accumulating"
            print(
                f"  [{i+1:4d}/{total}] ok    {sym:<6} {start}"
                f"  n_exp={cal.n_experiences:5d}  [{trained_str}]"
            )

        except KeyboardInterrupt:
            print("\n[train] interrupted, saving checkpoint...")
            break
        except Exception as e:
            print(f"  [{i+1:4d}/{total}] error {sym:<6} {start}  {e}")

    # 儲存
    cal.explore_std = 0.01   # 推論時小幅 exploration
    cal.save(args.out)
    print(f"\n[train] done  {ok_count}/{total} ok  n_exp={cal.n_experiences}")
    print(f"[train] calibrator saved -> {args.out}")
    print(f"        trained={cal.is_trained}")
    print(f"\n  下一步：")
    print(f"  python bench_generalize.py --runs 100")
    print(f"  （會自動載入 {args.out} 並在壓測中繼續學習）")


if __name__ == "__main__":
    main()
