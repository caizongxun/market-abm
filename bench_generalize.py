"""
bench_generalize.py
===================
泛化壓測腳本：隨機品種 × 隨機時段，跑 N 次 StatProcess rolling sim，
彙總統計矩誤差、DTW、path_corr，評估模型跨品種/跨時期泛化能力。

用法
----
  python bench_generalize.py                          # 預設 100 次
  python bench_generalize.py --runs 50 --workers 4   # 50 次，4 並行
  python bench_generalize.py --runs 20 --out results/bench_test.csv

參數
----
  --runs      總試驗次數（預設 100）
  --lookback  擬合視窗長度（預設 60）
  --step      滾動步進（預設 20）
  --seed      主隨機種子（預設 0）
  --workers   並行 worker 數（預設 1，建議不超過 CPU 核數）
  --out       結果 CSV 路徑（預設 results/bench_generalize.csv）
  --symbols   指定品種清單，空格分隔（預設內建清單）
  --min-days  最短歷史天數要求（預設 400）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 品種池（涵蓋美股科技、金融、能源、ETF、指數型 ETF）
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS = [
    # 科技大型股
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    # 金融
    "JPM", "GS", "BAC", "WFC", "MS",
    # 消費 / 醫療
    "WMT", "COST", "UNH", "JNJ", "PFE",
    # 能源 / 工業
    "XOM", "CVX", "CAT", "BA", "GE",
    # ETF
    "SPY", "QQQ", "IWM", "GLD", "TLT", "XLF", "XLE",
]

# 可用的歷史起始池（每筆至少 4 年前）
START_POOL = [
    "2018-01-01", "2018-07-01",
    "2019-01-01", "2019-07-01",
    "2020-01-01", "2020-07-01",
    "2021-01-01", "2021-07-01",
    "2022-01-01", "2022-07-01",
    "2023-01-01",
]


# ---------------------------------------------------------------------------
# 單次試驗
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    run_id:       int
    symbol:       str
    start:        str
    n_bars_total: int
    n_windows:    int
    # 模擬 vs 真實統計矩誤差（百分比絕對值）
    mean_err:     float   # |(μ_sim - μ_real) / σ_real|
    std_err_pct:  float   # |σ_sim/σ_real - 1|
    skew_err:     float   # |skew_sim - skew_real|
    kurt_err:     float   # |kurt_sim - kurt_real|
    hurst_err:    float   # |hurst_sim - hurst_real|
    dir_hit:      float   # 方向命中率
    dtw_mean:     float
    pcorr_mean:   float
    status:       str     # "ok" | "skip" | "error"
    error_msg:    str = ""


def _run_trial(
    run_id:   int,
    symbol:   str,
    start:    str,
    lookback: int,
    step:     int,
    seed:     int,
) -> TrialResult:
    """單一試驗，在子進程中執行。"""
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
                df_real  = df_real,
                lookback = lookback,
                step     = step,
                seed     = seed,
                verbose  = False,
            )

        n_sim = len(df_result)
        df_real_tail = df_real.iloc[-n_sim:].copy().reset_index(drop=True)

        r_real = np.diff(np.log(np.maximum(df_real_tail["Close"].values, 1e-10)))
        r_sim  = np.diff(np.log(np.maximum(df_result["Close"].values,   1e-10)))

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
        hurst_real  = float(hurst_exponent(r_real))
        hurst_sim   = float(hurst_exponent(r_sim))
        hurst_err   = abs(hurst_sim - hurst_real)

        # 方向命中率
        min_len  = min(len(r_real), len(r_sim))
        dir_hit  = float(np.mean(np.sign(r_real[:min_len]) == np.sign(r_sim[:min_len])))

        # DTW / path_corr（從 param_log _summary 提取）
        summary = next((p for p in param_log if p.get("_summary")), {})
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
            status="error", error_msg=str(e),
        )


# ---------------------------------------------------------------------------
# 報表列印
# ---------------------------------------------------------------------------

def _print_report(df: pd.DataFrame) -> None:
    ok = df[df["status"] == "ok"]
    skip_n  = int((df["status"] == "skip").sum())
    error_n = int((df["status"] == "error").sum())

    print("\n" + "=" * 62)
    print(f"  泛化壓測報告  ({len(df)} runs: {len(ok)} ok / {skip_n} skip / {error_n} error)")
    print("=" * 62)

    if len(ok) == 0:
        print("  沒有成功的試驗，請檢查品種/日期設定。")
        return

    metrics = [
        ("std_err_pct",  "std 誤差",      "< 0.10 ✅  (10%)"),
        ("skew_err",     "skew 誤差",     "< 0.50 ✅"),
        ("kurt_err",     "kurtosis 誤差", "< 3.00 ✅"),
        ("hurst_err",    "hurst 誤差",    "< 0.05 ✅"),
        ("dir_hit",      "方向命中率",     "> 0.52 ✅"),
        ("dtw_mean",     "DTW mean",      "< 0.05 ✅"),
        ("pcorr_mean",   "path_corr",     "> 0.10 ✅"),
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
    p = argparse.ArgumentParser(description="StatProcess generalization benchmark")
    p.add_argument("--runs",     type=int,   default=100)
    p.add_argument("--lookback", type=int,   default=60)
    p.add_argument("--step",     type=int,   default=20)
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--workers",  type=int,   default=1)
    p.add_argument("--out",      type=str,   default="results/bench_generalize.csv")
    p.add_argument("--symbols",  type=str,   nargs="*", default=None)
    p.add_argument("--min-days", type=int,   default=400)
    return p.parse_args()


def main():
    args   = parse_args()
    rng    = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS

    os.makedirs("results", exist_ok=True)

    # 生成試驗計劃
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
        # 並行
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for run_id, sym, start, seed in trials:
                fut = ex.submit(
                    _run_trial, run_id, sym, start,
                    args.lookback, args.step, seed
                )
                futures[fut] = run_id

            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                status_icon = "✓" if res.status == "ok" else "✗"
                print(
                    f"  [{res.run_id:3d}/{args.runs}] {status_icon}"
                    f" {res.symbol:<6} {res.start}"
                    f"  kurt_err={res.kurt_err:.2f}"
                    f"  dtw={res.dtw_mean:.4f}"
                    f"  status={res.status}"
                )
    else:
        # 循序
        for run_id, sym, start, seed in trials:
            res = _run_trial(run_id, sym, start, args.lookback, args.step, seed)
            results.append(res)
            status_icon = "✓" if res.status == "ok" else "✗"
            print(
                f"  [{run_id:3d}/{args.runs}] {status_icon}"
                f" {sym:<6} {start}"
                f"  kurt_err={'N/A' if np.isnan(res.kurt_err) else f'{res.kurt_err:.2f}'}"
                f"  dtw={'N/A' if np.isnan(res.dtw_mean) else f'{res.dtw_mean:.4f}'}"
                f"  dir_hit={'N/A' if np.isnan(res.dir_hit) else f'{res.dir_hit:.3f}'}"
                f"  status={res.status}"
            )

    df_out = pd.DataFrame([vars(r) for r in results])
    df_out.to_csv(args.out, index=False)
    print(f"\n[bench] results -> {args.out}")

    _print_report(df_out)
    print("\n[done]")


if __name__ == "__main__":
    main()
