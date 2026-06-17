"""
run_sim.py
==========
唯一入口。

基本用法
--------
  # ABM 模式（原有）
  python run_sim.py --symbol AAPL --bars 200 --plot
  python run_sim.py --symbol AAPL --start 2023-01-01 --rolling --plot

  # 統計過程模型（新）
  python run_sim.py --symbol AAPL --start 2023-01-01 --stat --plot
  python run_sim.py --symbol AAPL --start 2023-01-01 --stat --lookback 60 --step 20 --plot
  python run_sim.py --symbol AAPL --start 2023-01-01 --stat --lookback 60 --step 20 --conditional --plot

完整參數
--------
  --symbol          股票代碼（預設 AAPL）
  --bars            非 rolling 模式的模擬根數（預設 200）
  --start           歷史起始日期（rolling/stat 模式必填，例 2023-01-01）
  --seed            亂數種子（預設 42）
  --plot            產生比對圖
  --rolling         開啟 ABM rolling calibration 模式
  --stat            開啟統計過程模型 rolling 模式（Student-t + FBM + wick）
  --conditional     stat 模式下改用 block-bootstrap conditional generator（v70）
  --lookback        Rolling 模式的回看窗口（預設 60）
  --step            Rolling 模式的步進大小（預設 20）
  --ema-alpha       ABM rolling EMA 平滑係數（預設 0.4）
  --rolling-sims    ABM rolling 模式每組參數路徑數（預設 10）
  --w-vol           ABM rolling loss 中 vol_err 的權重（預設 1.5）
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from data.fetch import get_ohlcv
from sim.simulation import run_simulation
from sim.metrics import compare


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Market simulator (ABM + StatProcess)")
    p.add_argument("--symbol",       type=str,   default="AAPL")
    p.add_argument("--bars",         type=int,   default=200,
                   help="Bars to simulate (non-rolling mode)")
    p.add_argument("--start",        type=str,   default=None,
                   help="History start date (YYYY-MM-DD)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--plot",         action="store_true")
    # ABM rolling
    p.add_argument("--rolling",      action="store_true",
                   help="Enable ABM rolling-window calibration mode")
    p.add_argument("--lookback",     type=int,   default=60)
    p.add_argument("--step",         type=int,   default=20)
    p.add_argument("--ema-alpha",    type=float, default=0.4)
    p.add_argument("--rolling-sims", type=int,   default=10)
    p.add_argument("--w-vol",        type=float, default=1.5)
    # StatProcess
    p.add_argument("--stat",         action="store_true",
                   help="Enable StatProcess rolling mode (Student-t + FBM + wick)")
    p.add_argument("--conditional",  action="store_true",
                   help="Use block-bootstrap conditional generator (v70 Step2)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Static ABM mode
# ---------------------------------------------------------------------------
def run_static(args, df_real):
    print(f"[sim] {args.symbol}  bars={args.bars}  seed={args.seed}")
    _, df_sim = run_simulation(
        df_real=df_real,
        sim_bars=args.bars,
        seed=args.seed,
    )
    return df_sim


def save_results_static(symbol, df_sim):
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/{symbol}_sim.csv"
    df_sim.to_csv(out_csv, index=False)
    print(f"[out] -> {out_csv}")
    return out_csv


def plot_static(symbol, df_real, df_sim):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{symbol} \u2014 ABM Simulation vs Real", fontsize=13)

    axes[0, 0].plot(df_real["Close"].values[-len(df_sim):], label="Real", alpha=0.7)
    axes[0, 0].plot(df_sim["Close"].values,                 label="Sim",  alpha=0.7)
    axes[0, 0].set_title("Close Price")
    axes[0, 0].legend()

    r_real = df_real["Close"].pct_change().dropna()
    r_sim  = df_sim["Close"].pct_change().dropna()
    axes[0, 1].hist(r_real, bins=50, alpha=0.5, label="Real")
    axes[0, 1].hist(r_sim,  bins=50, alpha=0.5, label="Sim")
    axes[0, 1].set_title("Return Distribution")
    axes[0, 1].legend()

    axes[1, 0].plot(r_real.rolling(10).std().values, label="Real Vol",  alpha=0.7)
    axes[1, 0].plot(r_sim.rolling(10).std().values,  label="Sim Vol",   alpha=0.7)
    axes[1, 0].set_title("Rolling Volatility")
    axes[1, 0].legend()

    if "NetOrder" in df_sim.columns:
        axes[1, 1].bar(range(len(df_sim)), df_sim["NetOrder"].values, alpha=0.6)
        axes[1, 1].set_title("Net Order Flow")
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    out_png = f"results/{symbol}_comparison.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[plot] saved -> {out_png}")


# ---------------------------------------------------------------------------
# ABM Rolling mode
# ---------------------------------------------------------------------------
def run_rolling(args, df_real):
    from sim.regime import RegimeCalibrator
    import itertools

    cal = RegimeCalibrator(
        lookback  = args.lookback,
        step      = args.step,
        n_sims    = args.rolling_sims,
        ema_alpha = args.ema_alpha,
        w_vol     = args.w_vol,
        verbose   = True,
    )

    n_total = len(df_real)
    est_windows = (n_total - args.lookback) // args.step
    n_combos = len(list(itertools.product(*cal.grid.values())))
    est_runs = est_windows * n_combos * args.rolling_sims
    print(
        f"[rolling] {args.symbol}  total={n_total}  "
        f"lookback={args.lookback}  step={args.step}  "
        f"ema_alpha={args.ema_alpha}  rolling_sims={args.rolling_sims}"
    )
    print(f"[rolling] estimated windows: {est_windows}  (~{est_runs} sim runs)")

    df_result, param_log = cal.run(df_real, seed=args.seed)
    return df_result, param_log


def save_results_rolling(symbol, df_result, param_log):
    os.makedirs("results", exist_ok=True)

    out_csv = f"results/{symbol}_rolling_sim.csv"
    df_result.to_csv(out_csv, index=False)
    print(f"[out] -> {out_csv}")

    out_json = f"results/{symbol}_param_log.json"
    with open(out_json, "w") as f:
        json.dump(param_log, f, indent=2)
    print(f"[out] -> {out_json}  ({len(param_log)} windows)")

    return out_csv, out_json


def plot_rolling(symbol, df_real, df_result, param_log):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{symbol} \u2014 Rolling Calibration ABM", fontsize=13)

    n = len(df_result)
    axes[0, 0].plot(df_real["Close"].values[-n:], label="Real",  alpha=0.7)
    axes[0, 0].plot(df_result["Close"].values,    label="Sim",   alpha=0.7)
    axes[0, 0].set_title("Close Price")
    axes[0, 0].legend()

    windows     = [p["window"] for p in param_log]
    impact_vals = [p["impact_coeff"]      for p in param_log]
    scale_vals  = [p["momentum_scale"]    for p in param_log]
    noise_vals  = [p["intra_noise_scale"] for p in param_log]
    decay_vals  = [p["decay"]             for p in param_log]

    axes[0, 1].plot(windows, impact_vals, marker="o", ms=3, label="impact_coeff")
    axes[0, 1].set_title("impact_coeff (EMA smoothed)")

    ax2 = axes[1, 0]
    ax2.plot(windows, scale_vals, marker="o", ms=3, label="momentum_scale", color="tab:blue")
    ax2.set_ylabel("momentum_scale", color="tab:blue")
    ax2r = ax2.twinx()
    ax2r.plot(windows, noise_vals, marker="s", ms=3, label="intra_noise", color="tab:orange", alpha=0.7)
    ax2r.set_ylabel("intra_noise_scale", color="tab:orange")
    ax2.set_title("momentum_scale / intra_noise_scale")

    axes[1, 1].plot(windows, decay_vals, marker="o", ms=3, color="tab:green")
    axes[1, 1].set_title("decay (EMA smoothed)")
    axes[1, 1].set_ylim(0.85, 1.0)

    plt.tight_layout()
    out_png = f"results/{symbol}_rolling.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[plot] saved -> {out_png}")


# ---------------------------------------------------------------------------
# StatProcess rolling mode
# ---------------------------------------------------------------------------
def run_stat(args, df_real):
    from sim.stat_process import rolling_fit_generate

    n_total = len(df_real)
    est_windows = (n_total - args.lookback) // args.step
    print(
        f"[stat] {args.symbol}  total={n_total}  "
        f"lookback={args.lookback}  step={args.step}  seed={args.seed}"
    )
    print(f"[stat] estimated windows: {est_windows}")

    # v74: rolling_fit_generate now returns (df_result, param_log, ek_decay_ema)
    df_result, param_log, ek_decay_ema = rolling_fit_generate(
        df_real          = df_real,
        lookback         = args.lookback,
        step             = args.step,
        seed             = args.seed,
        verbose          = True,
        use_conditional  = args.conditional,
    )
    print(f"[stat] final ek_decay_ema={ek_decay_ema:.4f}")
    return df_result, param_log


def save_results_stat(symbol, df_result, param_log):
    os.makedirs("results", exist_ok=True)

    out_csv = f"results/{symbol}_stat_sim.csv"
    df_result.to_csv(out_csv, index=False)
    print(f"[out] -> {out_csv}")

    out_json = f"results/{symbol}_stat_params.json"
    with open(out_json, "w") as f:
        json.dump(param_log, f, indent=2, default=float)
    print(f"[out] -> {out_json}  ({len(param_log)} windows)")

    return out_csv, out_json


def plot_stat(symbol, df_real, df_result, param_log):
    """
    param_log keys (v70 stat_process):
      window, fit_start, fit_end,
      ret_std, ret_skew_a, hurst, target_ek,
      ek_oversample_adj, ek_oversample_init,
      jump_freq, vol_persistence,
      std_err_pct, kurt_err, hurst_err, dir_hit, c_err, use_conditional
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    cond_tag = " [conditional]" if any(p.get("use_conditional") for p in param_log if "window" in p) else ""
    fig.suptitle(f"{symbol} - StatProcess Rolling vs Real{cond_tag}", fontsize=13)

    n = len(df_result)
    real_tail = df_real.iloc[-n:].copy().reset_index(drop=True)

    # (0,0) Close price
    axes[0, 0].plot(real_tail["Close"].values, label="Real", alpha=0.7)
    axes[0, 0].plot(df_result["Close"].values,  label="Sim",  alpha=0.7)
    axes[0, 0].set_title("Close Price")
    axes[0, 0].legend()

    # (0,1) Return distribution
    r_real = np.diff(np.log(real_tail["Close"].values + 1e-10))
    r_sim  = np.diff(np.log(df_result["Close"].values + 1e-10))
    axes[0, 1].hist(r_real, bins=60, alpha=0.5, label="Real")
    axes[0, 1].hist(r_sim,  bins=60, alpha=0.5, label="Sim")
    axes[0, 1].set_title("Return Distribution")
    axes[0, 1].legend()

    # (0,2) Rolling Vol
    rv = pd.Series(r_real).rolling(10).std()
    sv = pd.Series(r_sim).rolling(10).std()
    axes[0, 2].plot(rv.values, label="Real Vol",  alpha=0.7)
    axes[0, 2].plot(sv.values, label="Sim Vol",   alpha=0.7)
    axes[0, 2].set_title("Rolling Volatility")
    axes[0, 2].legend()

    # filter to window records only (exclude _summary)
    window_params = [p for p in param_log if "window" in p]

    if not window_params:
        plt.tight_layout()
        out_png = f"results/{symbol}_stat.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[plot] saved -> {out_png}")
        return

    windows  = [p["window"]           for p in window_params]
    # v70 param_log uses 'hurst' (not 'hurst_target')
    hurst_v  = [p.get("hurst", p.get("hurst_target", 0.5))  for p in window_params]
    std_v    = [p.get("ret_std", 0.0)       for p in window_params]
    ek_v     = [p.get("target_ek", 0.0)     for p in window_params]
    ek_adj_v = [p.get("ek_oversample_adj", 1.0) for p in window_params]
    kurt_v   = [p.get("kurt_err") for p in window_params]
    kurt_v   = [v if v is not None and np.isfinite(v) else np.nan for v in kurt_v]

    # (1,0) target_ek per window
    ax = axes[1, 0]
    ax.plot(windows, ek_v, marker="o", ms=3, color="tab:red", label="target_ek")
    ax.set_title("target_ek per window")
    ax.axhline(y=3.0, linestyle="--", alpha=0.4, label="ek=3 (Gaussian)")
    ax.legend(fontsize=8)

    # (1,1) Hurst
    axes[1, 1].plot(windows, hurst_v, marker="o", ms=3, color="tab:blue")
    axes[1, 1].axhline(y=0.5, linestyle="--", alpha=0.4)
    axes[1, 1].set_title("Hurst Target")
    axes[1, 1].set_ylim(0.3, 0.8)

    # (1,2) ret_std + kurt_err
    ax = axes[1, 2]
    ax.plot(windows, std_v,  marker="o", ms=3, label="ret_std", color="tab:orange")
    ax.set_ylabel("ret_std", color="tab:orange")
    axr = ax.twinx()
    axr.plot(windows, kurt_v, marker="s", ms=3, label="kurt_err", color="tab:purple", alpha=0.7)
    axr.set_ylabel("kurt_err", color="tab:purple")
    ax.set_title("ret_std / kurt_err per window")

    plt.tight_layout()
    out_png = f"results/{symbol}_stat.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[plot] saved -> {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    df_real = get_ohlcv(args.symbol, start=args.start)

    # -----------------------------------------------------------------------
    if args.stat:
        df_result, param_log = run_stat(args, df_real)
        save_results_stat(args.symbol, df_result, param_log)

        n_sim = len(df_result)
        df_real_tail = df_real.iloc[-n_sim:].copy().reset_index(drop=True)
        print(f"\n[metrics] stat rolling sim vs. real ({n_sim} bars)")
        compare(df_real_tail, df_result, print_report=True)

        if args.plot:
            plot_stat(args.symbol, df_real, df_result, param_log)

    elif args.rolling:
        df_result, param_log = run_rolling(args, df_real)
        save_results_rolling(args.symbol, df_result, param_log)

        n_sim = len(df_result)
        df_real_tail = df_real.iloc[-n_sim:].copy().reset_index(drop=True)
        print(f"\n[metrics] rolling sim vs. real ({n_sim} bars)")
        compare(df_real_tail, df_result, print_report=True)

        if args.plot:
            plot_rolling(args.symbol, df_real, df_result, param_log)

    else:
        df_sim = run_static(args, df_real)
        save_results_static(args.symbol, df_sim)

        print(f"\n[metrics] sim vs. real ({args.bars} bars)")
        compare(
            df_real.iloc[-args.bars:].copy().reset_index(drop=True),
            df_sim,
            print_report=True,
        )

        if args.plot:
            plot_static(args.symbol, df_real, df_sim)

    print("\n[done]")


if __name__ == "__main__":
    main()
