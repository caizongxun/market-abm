"""
run_sim.py
==========
Main entry point: fetch data -> run simulation -> print stats -> (optional) plot.

Usage
-----
  python run_sim.py --symbol AAPL --bars 200 --plot
  python run_sim.py --symbol TSLA --bars 150 --warmup 120 --seed 0
  python run_sim.py --symbol SPY  --bars 200 --impact 0.0008 --noise 0.7 --plot
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data.fetch import fetch_ohlcv
from sim.simulation import run_simulation
from sim.metrics import compare


def parse_args():
    p = argparse.ArgumentParser(description="Agent-Based Market Simulation")
    p.add_argument("--symbol",  default="AAPL",  help="Ticker symbol")
    p.add_argument("--bars",    type=int, default=200, help="Number of bars to simulate")
    p.add_argument("--warmup",  type=int, default=100, help="Warm-up history bars")
    p.add_argument("--years",   type=int, default=3,   help="Years of history to download")
    p.add_argument("--impact",  type=float, default=0.0005, help="Market impact coefficient")
    p.add_argument("--noise",   type=float, default=0.6,    help="Intra-bar ATR noise scale")
    p.add_argument("--seed",    type=int,   default=42,     help="Random seed")
    p.add_argument("--plot",    action="store_true", help="Generate comparison plot")
    p.add_argument("--out-dir", default="results",   help="Output directory")
    # agent counts
    p.add_argument("--n-inst", type=int, default=5,   help="Institution agent count")
    p.add_argument("--n-mom",  type=int, default=40,  help="Momentum trader count")
    p.add_argument("--n-rand", type=int, default=100, help="Random trader count")
    p.add_argument("--n-cont", type=int, default=15,  help="Contrarian agent count")
    return p.parse_args()


def plot_comparison(
    df_warmup: pd.DataFrame,
    df_sim:    pd.DataFrame,
    df_real_future: pd.DataFrame,
    symbol:    str,
    out_path:  Path,
):
    """4-panel comparison plot: price / return dist / vol autocorr / net order."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"ABM vs. Real Market  |  {symbol}", fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#0f0f11")
    for ax in axes.flat:
        ax.set_facecolor("#18181c")
        ax.tick_params(colors="#71717a")
        ax.spines[:].set_color("#2a2a30")
        ax.xaxis.label.set_color("#71717a")
        ax.yaxis.label.set_color("#71717a")
        ax.title.set_color("#d4d4d8")

    # ── 1. Close Price ────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(df_warmup["Date"], df_warmup["Close"],  color="#6b7280", lw=1.2, label="History (warm-up)")
    ax.plot(df_sim["Date"],    df_sim["Close"],     color="#4f98a3", lw=1.2, label="Simulated")
    if not df_real_future.empty:
        ax.plot(df_real_future["Date"], df_real_future["Close"], color="#f0c040", lw=1.2, label="Real (same period)")
    ax.set_title("Close Price")
    ax.legend(fontsize=8, framealpha=0.3)

    # ── 2. Daily Return Distribution ────────────────────────────────────
    ax = axes[0, 1]
    real_rets = np.diff(np.log(df_real_future["Close"].values)) if not df_real_future.empty else np.array([])
    sim_rets  = np.diff(np.log(df_sim["Close"].values))
    bins = 60
    if len(real_rets) > 5:
        ax.hist(real_rets, bins=bins, density=True, alpha=0.55, color="#f0c040", label="Real")
    ax.hist(sim_rets, bins=bins, density=True, alpha=0.55, color="#4f98a3", label="Simulated")
    ax.set_title("Daily Return Distribution")
    ax.legend(fontsize=8, framealpha=0.3)

    # ── 3. Volatility Autocorrelation ──────────────────────────────────
    from sim.metrics import vol_autocorr
    ax = axes[1, 0]
    lags = range(1, 21)
    if len(real_rets) > 20:
        real_ac = vol_autocorr(real_rets, lags=20)
        ax.bar([l - 0.2 for l in lags], real_ac, width=0.35, color="#f0c040", alpha=0.7, label="Real")
    sim_ac = vol_autocorr(sim_rets, lags=20)
    ax.bar([l + 0.2 for l in lags], sim_ac, width=0.35, color="#4f98a3", alpha=0.7, label="Simulated")
    ax.axhline(0, color="#2a2a30", lw=0.8)
    ax.set_title("Volatility Autocorrelation (|ret|)")
    ax.set_xlabel("Lag")
    ax.legend(fontsize=8, framealpha=0.3)

    # ── 4. NetOrder (market sentiment proxy) ────────────────
    ax = axes[1, 1]
    ax.bar(range(len(df_sim)), df_sim["NetOrder"],
           color=np.where(df_sim["NetOrder"] > 0, "#26a69a", "#ef5350"),
           alpha=0.7)
    ax.axhline(0, color="#2a2a30", lw=0.8)
    ax.set_title("NetOrder (Buy/Sell Pressure)")
    ax.set_xlabel("Bar")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] saved -> {out_path}")


def print_ohlcv_comparison(df_real: pd.DataFrame, df_sim: pd.DataFrame, n: int = 10) -> None:
    """Print side-by-side OHLCV for the first/last n bars."""
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    real_cols = [c for c in cols if c in df_real.columns]
    sim_cols  = [c for c in cols if c in df_sim.columns]

    real_disp = df_real[real_cols].copy().reset_index(drop=True)
    sim_disp  = df_sim[sim_cols].copy().reset_index(drop=True)

    # align lengths
    length = min(len(real_disp), len(sim_disp))
    real_disp = real_disp.iloc[:length]
    sim_disp  = sim_disp.iloc[:length]

    # rename columns for side-by-side display
    real_disp.columns = [f"real_{c}" if c != "Date" else "Date" for c in real_disp.columns]
    sim_disp.columns  = [f"sim_{c}"  if c != "Date" else "Date" for c in sim_disp.columns]

    combined = pd.concat(
        [real_disp.add_suffix(""), sim_disp.drop(columns=["Date"], errors="ignore")],
        axis=1,
    )

    # show first n and last n rows
    display_n = min(n, length)
    head = combined.head(display_n)
    tail = combined.tail(display_n) if length > display_n else pd.DataFrame()

    float_cols = combined.select_dtypes(include="number").columns
    fmt = {c: "{:.4f}".format for c in float_cols if "Volume" not in c}
    vol_cols = [c for c in float_cols if "Volume" in c]
    fmt.update({c: "{:.0f}".format for c in vol_cols})

    sep = "-" * 120
    print(f"\n{'='*120}")
    print(f"  OHLCV Comparison  (real vs. simulated)  |  showing first/last {display_n} bars")
    print(f"{'='*120}")
    print("  --- First rows ---")
    print(head.to_string(index=True, formatters=fmt))
    if not tail.empty:
        print(f"\n  {sep}")
        print("  --- Last rows ---")
        print(tail.to_string(index=True, formatters=fmt))
    print(f"{'='*120}\n")

    # also print summary stats
    print("  Summary stats (Close)")
    stat_df = pd.DataFrame({
        "real_Close": df_real["Close"].describe() if "Close" in df_real.columns else pd.Series(dtype=float),
        "sim_Close":  df_sim["Close"].describe(),
    })
    print(stat_df.to_string())
    print()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch historical data
    end_date   = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=args.years * 365 + 60)).strftime("%Y-%m-%d")
    df_all = fetch_ohlcv(args.symbol, start_date, end_date)

    if len(df_all) < args.warmup + args.bars:
        print(f"[warn] Not enough data: need {args.warmup + args.bars} bars, got {len(df_all)}")
        print("       Increase --years or reduce --bars / --warmup")
        sys.exit(1)

    # Split: train (warm-up) vs real future (comparison target)
    df_train = df_all.iloc[:-(args.bars)].copy()
    df_real_future = df_all.iloc[-(args.bars):].copy().reset_index(drop=True)

    # 2. Run simulation
    print(f"\n[sim] {args.symbol}  warmup={args.warmup}  sim_bars={args.bars}  "
          f"impact={args.impact}  noise={args.noise}  seed={args.seed}")
    print(f"      agents: inst={args.n_inst}  mom={args.n_mom}  "
          f"rand={args.n_rand}  cont={args.n_cont}")

    df_warmup, df_sim = run_simulation(
        df_real=df_train,
        sim_bars=args.bars,
        warmup_bars=args.warmup,
        impact_coeff=args.impact,
        intra_noise_scale=args.noise,
        n_institution=args.n_inst,
        n_momentum=args.n_mom,
        n_random=args.n_rand,
        n_contrarian=args.n_cont,
        seed=args.seed,
    )

    # 3. Stats report
    print(f"\n[metrics] Comparing simulated vs. real ({args.bars} bars each)")
    report = compare(df_real_future, df_sim, print_report=True)

    # 4. OHLCV side-by-side comparison printout
    print_ohlcv_comparison(df_real_future, df_sim, n=10)

    # Save simulated bars
    sim_csv = out_dir / f"{args.symbol}_sim.csv"
    df_sim.to_csv(sim_csv, index=False)
    print(f"[out] Simulated bars -> {sim_csv}")

    # 5. Plot
    if args.plot:
        plot_comparison(
            df_warmup=df_warmup,
            df_sim=df_sim,
            df_real_future=df_real_future,
            symbol=args.symbol,
            out_path=out_dir / f"{args.symbol}_comparison.png",
        )

    print("\n[done]")


if __name__ == "__main__":
    main()
