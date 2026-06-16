"""
bench_generalize.py  v2
=======================
泛化壓測腳本：隨機品種 × 隨機時段，跑 N 次 StatProcess rolling sim，
彙總統計矩誤差、DTW、path_corr，評估模型跨品種/跨時期泛化能力。

v2 新增：
  - 跨 trial 共享 AdaptiveCalibrator 實例
  - 壓測結束後自動 save models/calibrator.pkl
  - 啟動時自動 load（若存在）
  - --no-calib 旗標可停用校正器（對照組）

用法
----
  python bench_generalize.py                          # 預設 100 次
  python bench_generalize.py --runs 50 --workers 4   # 50 次，4 並行
  python bench_generalize.py --no-calib              # 停用 calibrator
  python bench_generalize.py --runs 200 --out results/bench_v2.csv

參數
----
  --runs      總試驗次數（預設 100）
  --lookback  擬合視窗長度（預設 60）
  --step      滾動步進（預設 20）
  --seed      主隨機種子（預設 0）
  --workers   並行 worker 數（預設 1，calibrator 共享需 workers=1）
  --out       結果 CSV 路徑（預設 results/bench_generalize.csv）
  --symbols   指定品種清單，空格分隔（預設內建清單）
  --min-days  最短歷史天數要求（預設 400）
  --no-calib  停用 AdaptiveCalibrator，退化到 OnlineRidgePredictor
  --calib-path calibrator pkl 路徑（預設 models/calibrator.pkl）
"""
from __future__ import annotations

import argparse
import os
import random
import traceback
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 品種池
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# TrialResult
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    run_id:       int
    symbol:       str
    start:        str
    n_bars_total: int
    n_windows:    int
    mean_err:     float
    std_err_pct:  float
    skew_err:     float
    kurt_err:     float
    hurst_err:    float
    dir_hit:      float
    dtw_mean:     float
    pcorr_mean:   float
    status:       str
    error_msg:    str = ""


# ---------------------------------------------------------------------------
# 單次試驗（循序版，共享 calibrator）
# ---------------------------------------------------------------------------

def _run_trial_sequential(
    run_id:     int,
    symbol:     str,
    start:      str,
    lookback:   int,
    step:       int,
    seed:       int,
    calibrator,           # AdaptiveCalibrator | None
) -> TrialResult:
    try:
        from data.fetch import get_ohlcv
        from sim.stat_process import rolling_fit_generate
        from sim.metrics import hurst_exponent

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_real = get_ohlcv(symbol, start=start)

        if df_real is None or len(df_real) < lookback + step * 2:
            return TrialResult(
                run_id=run_id, symbol=symbol, start=start,
                n_bars_total=0, n_windows=0,
                mean_err=np.nan, std_err_pct=np.nan, skew_err=np.nan,
                kurt_err=np.nan, hurst_err=np.nan, dir_hit=np.nan,
                dtw_mean=np.nan, pcorr_mean=np.nan,
                status="skip", error_msg="insufficient bars",
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_result, param_log = rolling_fit_generate(
                df_real    = df_real,
                lookback   = lookback,
                step       = step,
                seed       = seed,
                verbose    = False,
                calibrator = calibrator,
            )

        n_sim        = len(df_result)
        df_real_tail = df_real.iloc[-n_sim:].copy().reset_index(drop=True)
        r_real = np.diff(np.log(np.maximum(df_real_tail["Close"].values, 1e-10)))
        r_sim  = np.diff(np.log(np.maximum(df_result["Close"].values,    1e-10)))

        if len(r_real) < 10 or len(r_sim) < 10:
            return TrialResult(
                run_id=run_id, symbol=symbol, start=start,
                n_bars_total=len(df_real), n_windows=0,
                mean_err=np.nan, std_err_pct=np.nan, skew_err=np.nan,
                kurt_err=np.nan, hurst_err=np.nan, dir_hit=np.nan,
                dtw_mean=np.nan, pcorr_mean=np.nan,
                status="skip", error_msg="too few returns",
            )

        real_std = float(np.std(r_real)) + 1e-10
        mean_err    = abs(float(np.mean(r_sim))  - float(np.mean(r_real)))  / real_std
        std_err_pct = abs(float(np.std(r_sim))   / real_std - 1.0)
        skew_err    = abs(float(stats.skew(r_sim))     - float(stats.skew(r_real)))
        kurt_err    = abs(float(stats.kurtosis(r_sim)) - float(stats.kurtosis(r_real)))
        hurst_err   = abs(float(hurst_exponent(r_sim)) - float(hurst_exponent(r_real)))
        min_len     = min(len(r_real), len(r_sim))
        dir_hit     = float(np.mean(np.sign(r_real[:min_len]) == np.sign(r_sim[:min_len])))

        summary    = next((p for p in param_log if p.get("_summary")), {})
        dtw_mean   = float(summary.get("dtw_mean",   np.nan) or np.nan)
        pcorr_mean = float(summary.get("pcorr_mean", np.nan) or np.nan)
        n_windows  = int(summary.get("n_windows", 0))

        return TrialResult(
            run_id=run_id, symbol=symbol, start=start,
            n_bars_total=len(df_real), n_windows=n_windows,
            mean_err=mean_err, std_err_pct=std_err_pct,
            skew_err=skew_err, kurt_err=kurt_err, hurst_err=hurst_err,
            dir_hit=dir_hit, dtw_mean=dtw_mean, pcorr_mean=pcorr_mean,
            status="ok",
        )

    except Exception as e:
        return TrialResult(
            run_id=run_id, symbol=symbol, start=start,
            n_bars_total=0, n_windows=0,
            mean_err=np.nan, std_err_pct=np.nan, skew_err=np.nan,
            kurt_err=np.nan, hurst_err=np.nan, dir_hit=np.nan,
            dtw_mean=np.nan, pcorr_mean=np.nan,
            status="error", error_msg=str(e)[:200],
        )


# ---------------------------------------------------------------------------
# 報表
# ---------------------------------------------------------------------------

def _print_report(df: pd.DataFrame, calib_n: int = 0) -> None:
    ok = df[df["status"] == "ok"]
    skip_n  = int((df["status"] == "skip").sum())
    error_n = int((df["status"] == "error").sum())

    print("\n" + "=" * 62)
    print(f"  泛化壓測報告  ({len(df)} runs: {len(ok)} ok / {skip_n} skip / {error_n} error)")
    if calib_n > 0:
        print(f"  AdaptiveCalibrator 累積經驗: {calib_n} 筆")
    print("=" * 62)

    if len(ok) == 0:
        print("  沒有成功的試驗。")
        return

    metrics = [
        ("std_err_pct",  "std 誤差",      "< 0.10"),
        ("skew_err",     "skew 誤差",     "< 0.50"),
        ("kurt_err",     "kurtosis 誤差", "< 3.00"),
        ("hurst_err",    "hurst 誤差",    "< 0.05"),
        ("dir_hit",      "方向命中率",     "> 0.52"),
        ("dtw_mean",     "DTW mean",      "< 0.90"),
        ("pcorr_mean",   "path_corr",     "> 0.10"),
    ]

    print(f"  {'指標':<16} {'中位數':>10} {'均值':>10} {'p10':>10} {'p90':>10}  目標")
    print("  " + "-" * 58)
    for col, label, target in metrics:
        vals = ok[col].dropna()
        if len(vals) == 0:
            print(f"  {label:<16}  {'N/A':>10}")
            continue
        print(
            f"  {label:<16}"
            f"  {vals.median():>10.4f}"
            f"  {vals.mean():>10.4f}"
            f"  {vals.quantile(0.10):>10.4f}"
            f"  {vals.quantile(0.90):>10.4f}"
            f"  {target}"
        )

    print("\n  按品種彙總 (ok runs only):")
    grp = ok.groupby("symbol")[["std_err_pct", "kurt_err", "dir_hit", "pcorr_mean"]].median()
    grp.columns = ["std_err%", "kurt_err", "dir_hit", "pcorr"]
    print(grp.to_string())
    print("=" * 62)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs",       type=int,   default=100)
    p.add_argument("--lookback",   type=int,   default=60)
    p.add_argument("--step",       type=int,   default=20)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--workers",    type=int,   default=1)
    p.add_argument("--out",        type=str,   default="results/bench_generalize.csv")
    p.add_argument("--symbols",    type=str,   nargs="*", default=None)
    p.add_argument("--min-days",   type=int,   default=400)
    p.add_argument("--no-calib",   action="store_true", help="停用 AdaptiveCalibrator")
    p.add_argument("--calib-path", type=str,   default="models/calibrator.pkl")
    return p.parse_args()


def main():
    args    = parse_args()
    rng     = random.Random(args.seed)
    np_rng  = np.random.default_rng(args.seed)
    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS

    os.makedirs("results", exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    # ---- 載入或初始化 calibrator ----
    calibrator = None
    if not args.no_calib:
        from sim.calibrator import AdaptiveCalibrator
        calibrator = AdaptiveCalibrator(min_train=50, update_interval=20, explore_std=0.03)
        if os.path.exists(args.calib_path):
            calibrator.load(args.calib_path)
            print(f"[calib] loaded {args.calib_path}  ({calibrator.n_experiences} experiences)")
        else:
            print(f"[calib] new calibrator (will save to {args.calib_path})")
    else:
        print("[calib] disabled")

    # ---- 試驗計劃 ----
    trials: list[tuple[int, str, str, int]] = []
    for i in range(args.runs):
        sym   = rng.choice(symbols)
        start = rng.choice(START_POOL)
        seed  = int(np_rng.integers(0, 2**31))
        trials.append((i + 1, sym, start, seed))

    print(f"[bench] {args.runs} runs  lookback={args.lookback}  step={args.step}  workers={args.workers}")
    print(f"[bench] symbols pool: {len(symbols)}  start pool: {len(START_POOL)}")

    results: list[TrialResult] = []

    if args.workers > 1:
        # 並行模式：calibrator 無法共享（跨進程），停用
        from concurrent.futures import ProcessPoolExecutor, as_completed
        if calibrator is not None:
            print("[calib] workers > 1: calibrator 停用（跨進程不共享）")
            calibrator = None

        def _run_trial_mp(args_tuple):
            run_id, sym, start, seed, lookback, step = args_tuple
            return _run_trial_sequential(run_id, sym, start, lookback, step, seed, None)

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_trial_mp, (rid, s, st, sd, args.lookback, args.step)): rid
                    for rid, s, st, sd in trials}
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                icon = "✓" if res.status == "ok" else "✗"
                print(f"  [{res.run_id:3d}/{args.runs}] {icon} {res.symbol:<6} {res.start}"
                      f"  kurt_err={res.kurt_err:.2f}  status={res.status}")
    else:
        # 循序模式：calibrator 跨 trial 共享
        for run_id, sym, start, seed in trials:
            res = _run_trial_sequential(run_id, sym, start,
                                        args.lookback, args.step, seed,
                                        calibrator)
            results.append(res)
            icon = "✓" if res.status == "ok" else "✗"
            calib_str = f"  calib_n={calibrator.n_experiences}" if calibrator else ""
            print(
                f"  [{run_id:3d}/{args.runs}] {icon} {sym:<6} {start}"
                f"  kurt_err={'N/A' if np.isnan(res.kurt_err) else f'{res.kurt_err:.2f}'}"
                f"  dtw={'N/A' if np.isnan(res.dtw_mean) else f'{res.dtw_mean:.4f}'}"
                f"  dir_hit={'N/A' if np.isnan(res.dir_hit) else f'{res.dir_hit:.3f}'}"
                f"  status={res.status}{calib_str}"
            )

    # ---- 儲存 calibrator ----
    if calibrator is not None:
        calibrator.explore_std = 0.0   # inference 時關閉 exploration
        calibrator.save(args.calib_path)
        print(f"[calib] saved {args.calib_path}  ({calibrator.n_experiences} experiences)")

    # ---- 儲存結果 & 報表 ----
    df_out = pd.DataFrame([vars(r) for r in results])
    df_out.to_csv(args.out, index=False)
    print(f"\n[bench] results -> {args.out}")

    calib_n = calibrator.n_experiences if calibrator else 0
    _print_report(df_out, calib_n=calib_n)
    print("\n[done]")


if __name__ == "__main__":
    main()
