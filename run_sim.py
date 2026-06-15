"""
run_sim.py — 唯一入口

Usage examples
--------------
  # 基本（無動能初始化）
  python run_sim.py --symbol AAPL --bars 60 --plot

  # 雙窗口動能初始化（建議預設）
  python run_sim.py --symbol AAPL --start 2024-01-01 --end 2025-06-01 \\
      --bars 60 --momentum-init --n-sims 200 --plot

  # 停用 auto_drift（對比實驗用）
  python run_sim.py --symbol AAPL --bars 60 --momentum-init --no-auto-drift --plot

  # 用 calibrator 找最佳靜態參數
  python -m sim.calibrate --symbol AAPL --start 2024-01-01 --end 2025-06-01

  # Rolling calibration（每段重新估參數）
  python run_sim.py --symbol AAPL --start 2023-01-01 --bars 200 \\
      --rolling --lookback 60 --step 20 --plot

Rolling calibration 說明
------------------------
--rolling        : 開啟 rolling mode
--lookback N     : 每個 window 用前 N 根 K 棒估參數（預設 60）
--step N         : 每隔 N 根 K 棒重新校準（預設 20）
--ema-alpha α    : EMA 平滑係數，0.4 = 新估值佔 40%（預設 0.4）
--rolling-sims N : 每組參數的路徑數（預設 10，越多越慢）

Decay guide (bias residual at bar N)
-------------------------------------
  decay=0.97  bar 20: 55%   bar 40: 30%   bar 60: 16%   <- default
  decay=0.95  bar 20: 36%   bar 40: 13%   bar 60:  5%
  decay=0.99  bar 20: 82%   bar 40: 67%   bar 60: 55%   <- 120+ bar sims
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data.fetch import fetch_ohlcv
from sim.simulation import run_simulation, run_simulation_rolling, estimate_momentum_drift_dual
from sim.metrics import compare


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Agent-Based Market Simulation")
    # Data
    p.add_argument("--symbol",  default="AAPL")
    p.add_argument("--start",   default=None)
    p.add_argument("--end",     default=None)
    p.add_argument("--years",   type=int, default=3)
    # Simulation
    p.add_argument("--bars",    type=int,   default=60)
    p.add_argument("--warmup",  type=int,   default=100)
    p.add_argument("--n-sims",  type=int,   default=1,
                   help="Paths to run. >1 generates fan chart.")
    p.add_argument("--impact",  type=float, default=0.0015)
    p.add_argument("--noise",   type=float, default=1.0)
    p.add_argument("--drift",   type=float, default=0.0)
    p.add_argument("--seed",    type=int,   default=42)
    # Momentum init
    p.add_argument("--momentum-init", action="store_true")
    p.add_argument("--no-auto-drift", action="store_true")
    p.add_argument("--momentum-window-fast", type=int,   default=5)
    p.add_argument("--momentum-window-slow", type=int,   default=20)
    p.add_argument("--momentum-scale",       type=float, default=1.0)
    p.add_argument("--decay",                type=float, default=0.97)
    # Price floor
    p.add_argument("--path-floor", type=float, default=0.30)
    # Agents
    p.add_argument("--n-inst",  type=int, default=5)
    p.add_argument("--n-mom",   type=int, default=40)
    p.add_argument("--n-rand",  type=int, default=100)
    p.add_argument("--n-cont",  type=int, default=15)
    # Output
    p.add_argument("--plot",    action="store_true")
    p.add_argument("--out-dir", default="results")
    # Rolling calibration
    p.add_argument("--rolling",       action="store_true",
                   help="Enable rolling calibration mode.")
    p.add_argument("--lookback",      type=int,   default=60,
                   help="Bars used to estimate params per window (default 60).")
    p.add_argument("--step",          type=int,   default=20,
                   help="Re-calibrate every N bars (default 20).")
    p.add_argument("--ema-alpha",     type=float, default=0.4,
                   help="EMA smoothing: 0.4 = new estimate 40%% (default 0.4).")
    p.add_argument("--rolling-sims",  type=int,   default=10,
                   help="Paths per combo in rolling grid search (default 10).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Multi-sim
# ---------------------------------------------------------------------------
def run_multi_sim(df_train, n_sims, args):
    auto_drift = not args.no_auto_drift
    paths = []
    for i in range(n_sims):
        _, df_sim = run_simulation(
            df_real=df_train,
            sim_bars=args.bars,
            warmup_bars=args.warmup,
            impact_coeff=args.impact,
            intra_noise_scale=args.noise,
            drift_per_bar=args.drift,
            momentum_window_fast=args.momentum_window_fast,
            momentum_window_slow=args.momentum_window_slow,
            momentum_scale=args.momentum_scale,
            bias_decay=args.decay,
            use_momentum_init=args.momentum_init,
            auto_drift=auto_drift,
            n_institution=args.n_inst,
            n_momentum=args.n_mom,
            n_random=args.n_rand,
            n_contrarian=args.n_cont,
            path_floor_pct=args.path_floor,
            seed=args.seed + i,
        )
        paths.append(df_sim)
    return paths


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _dark(fig, axes_flat):
    fig.patch.set_facecolor("#0f0f11")
    for ax in axes_flat:
        ax.set_facecolor("#18181c")
        ax.tick_params(colors="#71717a")
        ax.spines[:].set_color("#2a2a30")
        ax.xaxis.label.set_color("#71717a")
        ax.yaxis.label.set_color("#71717a")
        ax.title.set_color("#d4d4d8")


def plot_rolling(df_all, df_result, param_log, symbol, out_path):
    """Rolling mode 專用圖：價格 + 每個 window 的 impact / scale / decay 變化。"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 14),
                             gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    _dark(fig, axes)
    fig.suptitle(f"Rolling Calibration  |  {symbol}  |  "
                 f"{len(param_log)} windows  step={param_log[0]['bar_end']-param_log[0]['bar_start']}",
                 fontsize=12, fontweight="bold")

    # --- Price ---
    ax = axes[0]
    lookback = param_log[0]["bar_start"]
    df_ctx = df_all.iloc[:lookback]
    ax.plot(df_ctx["Date"], df_ctx["Close"], color="#6b7280", lw=1.0, label="History")
    ax.plot(df_result["Date"], df_result["Close"], color="#4f98a3", lw=1.2, label="Rolling sim")
    # Real (後半段)
    df_real_cmp = df_all.iloc[lookback:].reset_index(drop=True)
    if len(df_real_cmp) >= len(df_result):
        df_real_cmp = df_real_cmp.iloc[:len(df_result)]
    ax.plot(df_real_cmp["Date"], df_real_cmp["Close"],
            color="#f0c040", lw=1.2, label="Real", ls="--")
    # Window boundaries
    for pl in param_log:
        bar_idx = pl["bar_start"]
        if bar_idx < len(df_all):
            ax.axvline(df_all.iloc[bar_idx]["Date"], color="#3a3a45", lw=0.7, ls=":")
    ax.set_title("Close Price (history + rolling sim vs real)")
    ax.legend(fontsize=8, framealpha=0.3)

    # --- Param traces ---
    windows    = [pl["window"] + 1 for pl in param_log]
    impacts    = [pl["impact_coeff"] for pl in param_log]
    scales     = [pl["momentum_scale"] for pl in param_log]
    decays     = [pl["decay"] for pl in param_log]

    axes[1].plot(windows, impacts, color="#4f98a3", marker="o", ms=4, lw=1.5)
    axes[1].set_title("impact_coeff (EMA smoothed)")
    axes[1].set_ylabel("impact")

    axes[2].plot(windows, scales, color="#f97316", marker="o", ms=4, lw=1.5)
    axes[2].set_title("momentum_scale (EMA smoothed)")
    axes[2].set_ylabel("scale")

    axes[3].plot(windows, decays, color="#a78bfa", marker="o", ms=4, lw=1.5)
    axes[3].set_title("decay (EMA smoothed)")
    axes[3].set_ylabel("decay")
    axes[3].set_xlabel("Window")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] saved -> {out_path}")


def plot_fan_chart(df_warmup, paths, df_real_future, symbol, out_path):
    close_mat = np.vstack([p["Close"].values for p in paths])
    vol_mat   = np.vstack([p["Volume"].values for p in paths])
    med       = np.median(close_mat, axis=0)
    p10       = np.percentile(close_mat, 10, axis=0)
    p90       = np.percentile(close_mat, 90, axis=0)
    hv_idx    = int(np.argmax(vol_mat.sum(axis=1)))
    hv_path   = paths[hv_idx]
    sim_dates = paths[0]["Date"].values
    n_sims    = len(paths)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    _dark(fig, axes.flat)
    fig.suptitle(f"ABM Fan Chart  |  {symbol}  |  {n_sims} paths",
                 fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df_warmup["Date"], df_warmup["Close"], color="#6b7280", lw=1.0, label="History")
    for p in paths:
        ax.plot(sim_dates, p["Close"].values, color="#4f98a3", alpha=0.07, lw=0.5)
    ax.fill_between(sim_dates, p10, p90, color="#4f98a3", alpha=0.20, label="10-90th pct")
    ax.plot(sim_dates, med, color="#4f98a3", lw=2.0, label="Median sim")
    ax.plot(sim_dates, hv_path["Close"].values, color="#f97316",
            lw=1.4, ls="--", label=f"Highest-vol #{hv_idx}")
    if not df_real_future.empty:
        ax.plot(df_real_future["Date"], df_real_future["Close"],
                color="#f0c040", lw=2.0, label="Real")
    ax.set_title("Close Price Fan Chart")
    ax.legend(fontsize=8, framealpha=0.3)

    ax = axes[0, 1]
    all_rets = np.concatenate([np.diff(np.log(p["Close"].values)) for p in paths])
    real_rets = (np.diff(np.log(df_real_future["Close"].values))
                 if not df_real_future.empty else np.array([]))
    if len(real_rets) > 5:
        ax.hist(real_rets, bins=60, density=True, alpha=0.55, color="#f0c040", label="Real")
    ax.hist(all_rets, bins=60, density=True, alpha=0.45, color="#4f98a3",
            label=f"Sim ({n_sims} paths)")
    ax.set_title("Return Distribution")
    ax.legend(fontsize=8, framealpha=0.3)

    ax = axes[1, 0]
    dir_mat  = (close_mat[:, 1:] > close_mat[:, :-1]).astype(float)
    prob_up  = dir_mat.mean(axis=0)
    bar_x    = np.arange(len(prob_up))
    ax.bar(bar_x, prob_up - 0.5,
           color=np.where(prob_up >= 0.5, "#26a69a", "#ef5350"), alpha=0.8)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    if not df_real_future.empty and len(df_real_future) > 1:
        real_dir = (df_real_future["Close"].values[1:]
                    > df_real_future["Close"].values[:-1]).astype(float)
        n_show = min(len(real_dir), len(bar_x))
        ax.scatter(bar_x[:n_show], real_dir[:n_show] - 0.5,
                   color="#f0c040", s=18, zorder=5, label="Real direction")
        ax.legend(fontsize=8, framealpha=0.3)
    ax.set_title("P(Up) per Bar  (above 0 = majority bullish)")
    ax.set_xlabel("Bar")

    ax = axes[1, 1]
    if not df_real_future.empty:
        rn = df_real_future["Close"].values / df_real_future["Close"].values[0] * 100
        ax.plot(range(len(rn)), rn, color="#f0c040", lw=2.0, label="Real (indexed)")
    mn = med / med[0] * 100
    ax.plot(range(len(mn)), mn, color="#4f98a3", lw=2.0, label="Median sim (indexed)")
    ax.fill_between(range(len(mn)),
                    p10 / med[0] * 100, p90 / med[0] * 100,
                    color="#4f98a3", alpha=0.15)
    ax.axhline(100, color="#6b7280", lw=0.6, ls=":")
    ax.set_title("Indexed Performance (start = 100)")
    ax.legend(fontsize=8, framealpha=0.3)
    ax.set_xlabel("Bar")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] saved -> {out_path}")


def plot_single(df_warmup, df_sim, df_real_future, symbol, out_path):
    from sim.metrics import vol_autocorr
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _dark(fig, axes.flat)
    fig.suptitle(f"ABM vs. Real  |  {symbol}", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df_warmup["Date"], df_warmup["Close"], color="#6b7280", lw=1.2, label="History")
    ax.plot(df_sim["Date"],    df_sim["Close"],    color="#4f98a3", lw=1.2, label="Simulated")
    if not df_real_future.empty:
        ax.plot(df_real_future["Date"], df_real_future["Close"],
                color="#f0c040", lw=1.2, label="Real")
    ax.set_title("Close Price")
    ax.legend(fontsize=8, framealpha=0.3)

    ax = axes[0, 1]
    real_rets = (np.diff(np.log(df_real_future["Close"].values))
                 if not df_real_future.empty else np.array([]))
    sim_rets  = np.diff(np.log(df_sim["Close"].values))
    if len(real_rets) > 5:
        ax.hist(real_rets, bins=60, density=True, alpha=0.55, color="#f0c040", label="Real")
    ax.hist(sim_rets, bins=60, density=True, alpha=0.55, color="#4f98a3", label="Simulated")
    ax.set_title("Return Distribution")
    ax.legend(fontsize=8, framealpha=0.3)

    ax = axes[1, 0]
    lags = range(1, 21)
    if len(real_rets) > 20:
        real_ac = vol_autocorr(real_rets, lags=20)
        ax.bar([l - 0.2 for l in lags], real_ac, width=0.35,
               color="#f0c040", alpha=0.7, label="Real")
    sim_ac = vol_autocorr(sim_rets, lags=20)
    ax.bar([l + 0.2 for l in lags], sim_ac, width=0.35,
           color="#4f98a3", alpha=0.7, label="Simulated")
    ax.axhline(0, color="#2a2a30", lw=0.8)
    ax.set_title("Volatility Autocorrelation")
    ax.legend(fontsize=8, framealpha=0.3)

    ax = axes[1, 1]
    ax.bar(range(len(df_sim)), df_sim["NetOrder"],
           color=np.where(df_sim["NetOrder"] > 0, "#26a69a", "#ef5350"), alpha=0.7)
    ax.axhline(0, color="#2a2a30", lw=0.8)
    ax.set_title("NetOrder")
    ax.set_xlabel("Bar")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] saved -> {out_path}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def print_ohlcv_comparison(df_real, df_sim, n=10):
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    real = df_real[[c for c in cols if c in df_real.columns]].copy().reset_index(drop=True)
    sim  = df_sim [[c for c in cols if c in df_sim.columns ]].copy().reset_index(drop=True)
    length = min(len(real), len(sim))
    real, sim = real.iloc[:length], sim.iloc[:length]
    real.columns = [f"real_{c}" if c != "Date" else "Date" for c in real.columns]
    sim.columns  = [f"sim_{c}"  if c != "Date" else "Date" for c in sim.columns]
    combined = pd.concat([real, sim.drop(columns=["Date"], errors="ignore")], axis=1)
    dn   = min(n, length)
    head = combined.head(dn)
    tail = combined.tail(dn) if length > dn else pd.DataFrame()
    fc   = combined.select_dtypes(include="number").columns
    fmt  = {c: "{:.4f}".format for c in fc if "Volume" not in c}
    fmt.update({c: "{:.0f}".format for c in fc if "Volume" in c})
    print(f"\n{'='*120}")
    print(f"  OHLCV Comparison  |  first/last {dn} bars")
    print(f"{'='*120}")
    print(head.to_string(index=True, formatters=fmt))
    if not tail.empty:
        print(f"\n  {'-'*120}")
        print(tail.to_string(index=True, formatters=fmt))
    print(f"{'='*120}\n")
    stat = pd.DataFrame({"real_Close": df_real["Close"].describe(),
                          "sim_Close":  df_sim["Close"].describe()})
    print("  Summary stats (Close)")
    print(stat.to_string())
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    auto_drift = not args.no_auto_drift
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch
    end_date   = args.end   or date.today().strftime("%Y-%m-%d")
    start_date = args.start or (
        date.today() - timedelta(days=args.years * 365 + 60)
    ).strftime("%Y-%m-%d")
    print(f"[fetch] range: {start_date} ~ {end_date}")
    df_all = fetch_ohlcv(args.symbol, start_date, end_date)

    # -----------------------------------------------------------------------
    # Rolling calibration mode
    # -----------------------------------------------------------------------
    if args.rolling:
        min_needed = args.lookback + args.step
        if len(df_all) < min_needed:
            print(f"[error] rolling mode needs >= {min_needed} bars, got {len(df_all)}")
            sys.exit(1)

        print(f"[rolling] {args.symbol}  total={len(df_all)}  "
              f"lookback={args.lookback}  step={args.step}  "
              f"ema_alpha={args.ema_alpha}  rolling_sims={args.rolling_sims}")
        n_windows = (len(df_all) - args.lookback) // args.step
        print(f"[rolling] estimated windows: {n_windows}  "
              f"(~{n_windows * 27 * args.rolling_sims} sim runs)")

        df_result, param_log = run_simulation_rolling(
            df_all=df_all,
            lookback=args.lookback,
            step=args.step,
            n_sims=args.rolling_sims,
            warmup_bars=min(args.warmup, args.lookback),
            ema_alpha=args.ema_alpha,
            seed=args.seed,
            verbose=True,
        )

        # Save outputs
        out_csv = out_dir / f"{args.symbol}_rolling_sim.csv"
        df_result.to_csv(out_csv, index=False)
        print(f"[out] -> {out_csv}")

        import json
        out_log = out_dir / f"{args.symbol}_param_log.json"
        with open(out_log, "w") as f:
            json.dump(param_log, f, indent=2, default=float)
        print(f"[out] -> {out_log}  ({len(param_log)} windows)")

        # Metrics vs real (後半段)
        df_real_cmp = df_all.iloc[args.lookback:].reset_index(drop=True)
        n_cmp = min(len(df_real_cmp), len(df_result))
        if n_cmp > 5:
            print(f"\n[metrics] rolling sim vs. real ({n_cmp} bars)")
            compare(df_real_cmp.iloc[:n_cmp], df_result.iloc[:n_cmp], print_report=True)

        if args.plot:
            plot_rolling(
                df_all, df_result, param_log, args.symbol,
                out_dir / f"{args.symbol}_rolling.png",
            )
        print("\n[done]")
        return

    # -----------------------------------------------------------------------
    # Static mode (original)
    # -----------------------------------------------------------------------
    if len(df_all) < args.warmup + args.bars:
        print(f"[warn] Need {args.warmup + args.bars} bars, got {len(df_all)}")
        sys.exit(1)

    df_train       = df_all.iloc[:-(args.bars)].copy()
    df_real_future = df_all.iloc[-(args.bars):].copy().reset_index(drop=True)
    print(f"[data] train rows: {len(df_train)}  "
          f"real comparison rows: {len(df_real_future)}  "
          f"({df_real_future['Date'].iloc[0].date()} ~ "
          f"{df_real_future['Date'].iloc[-1].date()})")

    if args.momentum_init:
        ctx_closes = df_train.tail(args.warmup)["Close"].values.astype(float)
        mom_drift, reason = estimate_momentum_drift_dual(
            ctx_closes,
            window_fast=args.momentum_window_fast,
            window_slow=args.momentum_window_slow,
            scale=args.momentum_scale,
        )
        direction = "UP" if mom_drift > 0 else ("DOWN" if mom_drift < 0 else "FLAT")
        half_life = -np.log(2) / np.log(args.decay) if args.decay < 1.0 else float("inf")
        residual_at_end = args.decay ** args.bars * 100
        print(f"[momentum-init] {reason}")
        print(f"               => {direction} bias  "
              f"{mom_drift:+.6f}/bar  "
              f"(annualised ~{mom_drift * 252:.2%})  "
              f"decay={args.decay}  "
              f"half-life={half_life:.1f} bars  "
              f"residual@bar{args.bars}={residual_at_end:.1f}%")
        if auto_drift:
            print(f"               [auto_drift=ON] bias routed to drift_per_bar schedule")
        else:
            print(f"               [auto_drift=OFF] bias routed through agent orders")

    print(f"\n[sim] {args.symbol}  warmup={args.warmup}  bars={args.bars}  "
          f"n_sims={args.n_sims}  impact={args.impact}  noise={args.noise}  "
          f"seed={args.seed}  path_floor={args.path_floor:.0%}")
    print(f"      agents: inst={args.n_inst}  mom={args.n_mom}  "
          f"rand={args.n_rand}  cont={args.n_cont}")

    sim_kwargs = dict(
        sim_bars=args.bars,
        warmup_bars=args.warmup,
        impact_coeff=args.impact,
        intra_noise_scale=args.noise,
        drift_per_bar=args.drift,
        momentum_window_fast=args.momentum_window_fast,
        momentum_window_slow=args.momentum_window_slow,
        momentum_scale=args.momentum_scale,
        bias_decay=args.decay,
        use_momentum_init=args.momentum_init,
        auto_drift=auto_drift,
        n_institution=args.n_inst,
        n_momentum=args.n_mom,
        n_random=args.n_rand,
        n_contrarian=args.n_cont,
        path_floor_pct=args.path_floor,
    )

    if args.n_sims == 1:
        df_warmup, df_sim = run_simulation(
            df_real=df_train, seed=args.seed, **sim_kwargs
        )
        print(f"\n[metrics] sim vs. real ({args.bars} bars)")
        compare(df_real_future, df_sim, print_report=True)
        print_ohlcv_comparison(df_real_future, df_sim, n=10)
        df_sim.to_csv(out_dir / f"{args.symbol}_sim.csv", index=False)
        print(f"[out] -> {out_dir}/{args.symbol}_sim.csv")
        if args.plot:
            plot_single(df_warmup, df_sim, df_real_future, args.symbol,
                        out_dir / f"{args.symbol}_comparison.png")
    else:
        print(f"[sim] Running {args.n_sims} paths...")
        paths = []
        for i in range(args.n_sims):
            _, df_sim = run_simulation(
                df_real=df_train, seed=args.seed + i, **sim_kwargs
            )
            paths.append(df_sim)
        df_warmup = df_train.tail(args.warmup).reset_index(drop=True)

        close_mat  = np.vstack([p["Close"].values for p in paths])
        vol_mat    = np.vstack([p["Volume"].values for p in paths])
        final_rets = (close_mat[:, -1] - close_mat[:, 0]) / close_mat[:, 0]
        prob_up    = float((final_rets > 0).mean())
        med_ret    = float(np.median(final_rets))
        p10_ret    = float(np.percentile(final_rets, 10))
        p90_ret    = float(np.percentile(final_rets, 90))
        hv_idx     = int(np.argmax(vol_mat.sum(axis=1)))
        hv_ret     = float(
            (paths[hv_idx]["Close"].iloc[-1] - paths[hv_idx]["Close"].iloc[0])
            / paths[hv_idx]["Close"].iloc[0]
        )
        real_ret = (
            float((df_real_future["Close"].iloc[-1] - df_real_future["Close"].iloc[0])
                  / df_real_future["Close"].iloc[0])
            if not df_real_future.empty else float("nan")
        )

        print(f"\n[multi-sim]  {args.n_sims} paths  /  {args.bars} bars")
        print(f"  P(final close > start)  : {prob_up:.1%}")
        print(f"  Median total return     : {med_ret:+.2%}")
        print(f"  10th / 90th pct return  : {p10_ret:+.2%} / {p90_ret:+.2%}")
        print(f"  Highest-vol path #{hv_idx:<3}   : {hv_ret:+.2%}")
        if not np.isnan(real_ret):
            print(f"  Real total return       : {real_ret:+.2%}")
            in_band = p10_ret <= real_ret <= p90_ret
            match   = (prob_up >= 0.5) == (real_ret > 0)
            print(f"  Direction match         : {'YES' if match else 'NO'}")
            print(f"  Real in 10-90th band    : {'YES' if in_band else 'NO'}")

        med_close = np.median(close_mat, axis=0)
        df_med    = paths[0].copy()
        df_med["Close"] = med_close
        df_med.to_csv(out_dir / f"{args.symbol}_sim_median.csv", index=False)
        paths[hv_idx].to_csv(out_dir / f"{args.symbol}_sim_highvol.csv", index=False)
        print(f"[out] -> {out_dir}/{args.symbol}_sim_median.csv")
        print(f"[out] -> {out_dir}/{args.symbol}_sim_highvol.csv")

        if not df_real_future.empty:
            print(f"\n[metrics] median path vs. real ({args.bars} bars)")
            compare(df_real_future, df_med, print_report=True)

        if args.plot:
            plot_fan_chart(df_warmup, paths, df_real_future, args.symbol,
                           out_dir / f"{args.symbol}_fan_chart.png")

    print("\n[done]")


if __name__ == "__main__":
    main()
