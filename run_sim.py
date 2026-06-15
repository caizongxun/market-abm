"""
run_sim.py
==========
唯一入口：下載資料 → 跑模擬 → 印統計報告 → （可選）畫圖。

用法
----
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

# プロジェクトルートを path に追加
sys.path.insert(0, str(Path(__file__).parent))

from data.fetch import fetch_ohlcv
from sim.simulation import run_simulation
from sim.metrics import compare


def parse_args():
    p = argparse.ArgumentParser(description="Agent-Based Market Simulation")
    p.add_argument("--symbol",  default="AAPL",  help="股票代碼")
    p.add_argument("--bars",    type=int, default=200, help="模擬根數")
    p.add_argument("--warmup",  type=int, default=100, help="warm-up 歷史根數")
    p.add_argument("--years",   type=int, default=3,   help="下載歷史年數")
    p.add_argument("--impact",  type=float, default=0.0005, help="市場衝擊係數")
    p.add_argument("--noise",   type=float, default=0.6,    help="K棒內波動 ATR 倍數")
    p.add_argument("--seed",    type=int,   default=42,     help="隨機種子")
    p.add_argument("--plot",    action="store_true", help="產生比對圖")
    p.add_argument("--out-dir", default="results",   help="輸出目錄")
    # agent 數量
    p.add_argument("--n-inst", type=int, default=5,   help="機構 agent 數")
    p.add_argument("--n-mom",  type=int, default=40,  help="動能散戶數")
    p.add_argument("--n-rand", type=int, default=100, help="隨機散戶數")
    p.add_argument("--n-cont", type=int, default=15,  help="逆勢 agent 數")
    return p.parse_args()


def plot_comparison(
    df_warmup: pd.DataFrame,
    df_sim:    pd.DataFrame,
    df_real_future: pd.DataFrame,
    symbol:    str,
    out_path:  Path,
):
    """畫 4 格比對圖：K 棒走勢 / 報酬分佈 / Hurst / 波動率自相關。"""
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

    # ── 1. 收盤價走勢 ────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(df_warmup["Date"], df_warmup["Close"],  color="#6b7280", lw=1.2, label="歷史（warm-up）")
    ax.plot(df_sim["Date"],    df_sim["Close"],     color="#4f98a3", lw=1.2, label="模擬")
    if not df_real_future.empty:
        ax.plot(df_real_future["Date"], df_real_future["Close"], color="#f0c040", lw=1.2, label="真實（同期）")
    ax.set_title("收盤價走勢")
    ax.legend(fontsize=8, framealpha=0.3)

    # ── 2. 日報酬分佈 ────────────────────────────────────
    ax = axes[0, 1]
    real_rets = np.diff(np.log(df_real_future["Close"].values)) if not df_real_future.empty else np.array([])
    sim_rets  = np.diff(np.log(df_sim["Close"].values))
    bins = 60
    if len(real_rets) > 5:
        ax.hist(real_rets, bins=bins, density=True, alpha=0.55, color="#f0c040", label="真實")
    ax.hist(sim_rets, bins=bins, density=True, alpha=0.55, color="#4f98a3", label="模擬")
    ax.set_title("日報酬分佈")
    ax.legend(fontsize=8, framealpha=0.3)

    # ── 3. 波動率自相關 ──────────────────────────────────
    from sim.metrics import vol_autocorr
    ax = axes[1, 0]
    lags = range(1, 21)
    if len(real_rets) > 20:
        real_ac = vol_autocorr(real_rets, lags=20)
        ax.bar([l - 0.2 for l in lags], real_ac, width=0.35, color="#f0c040", alpha=0.7, label="真實")
    sim_ac = vol_autocorr(sim_rets, lags=20)
    ax.bar([l + 0.2 for l in lags], sim_ac, width=0.35, color="#4f98a3", alpha=0.7, label="模擬")
    ax.axhline(0, color="#2a2a30", lw=0.8)
    ax.set_title("波動率自相關（|ret| autocorr）")
    ax.set_xlabel("Lag")
    ax.legend(fontsize=8, framealpha=0.3)

    # ── 4. NetOrder 走勢（市場情緒代理）────────────────
    ax = axes[1, 1]
    ax.bar(range(len(df_sim)), df_sim["NetOrder"],
           color=np.where(df_sim["NetOrder"] > 0, "#26a69a", "#ef5350"),
           alpha=0.7)
    ax.axhline(0, color="#2a2a30", lw=0.8)
    ax.set_title("NetOrder（淨買賣壓）")
    ax.set_xlabel("Bar")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] saved → {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 下載歷史資料
    end_date   = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=args.years * 365 + 60)).strftime("%Y-%m-%d")
    df_all = fetch_ohlcv(args.symbol, start_date, end_date)

    if len(df_all) < args.warmup + args.bars:
        print(f"[warn] 資料不足：需要 {args.warmup + args.bars} 根，實際 {len(df_all)} 根")
        print("       請增加 --years 或減少 --bars / --warmup")
        sys.exit(1)

    # 使用前段作為 warm-up + 模擬對照，後段作為真實對照
    df_train = df_all.iloc[:-(args.bars)].copy()
    df_real_future = df_all.iloc[-(args.bars):].copy().reset_index(drop=True)

    # 2. 執行模擬
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

    # 3. 統計報告
    print(f"\n[metrics] 比較模擬 vs. 真實（各 {args.bars} 根）")
    report = compare(df_real_future, df_sim, print_report=True)

    # 儲存模擬結果
    sim_csv = out_dir / f"{args.symbol}_sim.csv"
    df_sim.to_csv(sim_csv, index=False)
    print(f"[out] 模擬 K 棒 → {sim_csv}")

    # 4. 畫圖
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
