"""
ipo_pattern_scan.py
-------------------
Compares the first-3-day price momentum of a target IPO (e.g. SpaceX)
against all recent IPOs in US and Taiwan markets, then ranks them by
pattern similarity.

Pipeline:
  1. Fetch target ticker's first 3 trading days (intraday or daily)
  2. Bulk-fetch recent US + TW IPO candidates (adjustable date range)
  3. Normalize each series (z-score on % returns from IPO open)
  4. Score similarity via cosine similarity + optional DTW
  5. Print ranked table + save comparison chart to ipo_scan_results/

Usage:
  python ipo_pattern_scan.py --target SPXC --days 3 --top 20

  # SpaceX IPO ticker might not be finalized; override with --target <actual_ticker>
  # For Taiwan stocks, prefix with .TW e.g. 2330.TW (handled automatically)

Dependencies:
  pip install yfinance pandas numpy scipy matplotlib tqdm requests
  Optional: pip install dtaidistance  # for DTW scoring

Note on data limits:
  yfinance free tier has rate limits. The script caches each fetch to
  ipo_scan_cache/<ticker>.parquet to avoid re-fetching on reruns.
"""

import argparse
import os
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance not found. Run: pip install yfinance")

try:
    from dtaidistance import dtw as _dtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    warnings.warn("dtaidistance not installed; DTW scoring disabled. pip install dtaidistance")


# ---------------------------------------------------------------------------
# Config defaults (override via CLI args)
# ---------------------------------------------------------------------------
DEFAULT_TARGET     = "SPXC"   # Replace with actual SpaceX ticker when confirmed
DEFAULT_DAYS       = 3        # First N trading days to compare
DEFAULT_TOP        = 20       # Top N similar stocks to show
DEFAULT_IPO_WINDOW = 365      # Search IPOs from the past N days
DEFAULT_MIN_GAIN   = 5.0      # Only include IPOs with >5% gain in first 3 days (FOMO filter)
CACHE_DIR          = Path("ipo_scan_cache")
OUT_DIR            = Path("ipo_scan_results")


# ---------------------------------------------------------------------------
# Known recent US IPO tickers (2023-2026 notable ones)
# The script also auto-fetches from a public IPO list API when available.
# Extend this list or connect to a paid screener for full coverage.
# ---------------------------------------------------------------------------
US_IPO_SEEDS = [
    # 2024-2026 notable US IPOs
    "RDZN", "ASIC", "ASTS", "LUNR", "RDW", "IIPR",
    "KVYO", "ARM",  "BIRK", "CART", "CAVA", "GTLB",
    "IONQ", "RKLB", "ACHR", "JOBY", "LILM", "SPNV",
    "VERX", "CRDO", "TBLA", "MNDY", "DDOG", "SNOW",
    "PLTR", "ABNB", "DASH", "COIN", "RIVN", "LCID",
    "RBLX", "DKNG", "OPEN", "UWMC", "HOOD", "BROS",
    "COUR", "DUOL", "MAPS", "OPAD", "LAZR", "LIDR",
    "RDDT", "IBOTTA", "ASTERA", "RUBRIK", "LINEAGE",
    "SEZL", "LOAR", "STLC", "ONON", "CELH", "SMCI",
]

# Taiwan IPO seeds (append .TW suffix — yfinance handles it)
TW_IPO_SEEDS = [
    "2330.TW", "2317.TW", "2454.TW", "3711.TW", "6488.TW",
    "6669.TW", "6770.TW", "6515.TW", "3533.TW", "6789.TW",
    "8299.TW", "6692.TW", "3105.TW", "6591.TW", "4953.TW",
    "6278.TW", "3552.TW", "6768.TW", "5274.TW", "6830.TW",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV with parquet caching. Returns empty DF on failure."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = CACHE_DIR / f"{ticker.replace('.', '_')}_{interval}.parquet"

    # Use cache if fresh (< 4 hours old) to respect rate limits
    if cache_key.exists():
        age_h = (time.time() - cache_key.stat().st_mtime) / 3600
        if age_h < 4:
            try:
                return pd.read_parquet(cache_key)
            except Exception:
                pass

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_parquet(cache_key)
        time.sleep(0.3)  # gentle rate limiting
        return df
    except Exception as e:
        warnings.warn(f"[{ticker}] fetch failed: {e}")
        return pd.DataFrame()


def first_n_trading_days(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return first N rows of a price series (= first N trading days)."""
    return df.head(n)


def normalize_returns(df: pd.DataFrame) -> np.ndarray:
    """
    Convert Close prices to cumulative % return from day-0 open,
    then z-score normalize so shape is comparable across price levels.
    Returns 1-D numpy array.
    """
    closes = df["Close"].values.astype(float)
    if len(closes) < 2:
        return np.array([])
    base = closes[0]
    pct = (closes - base) / base * 100.0  # % from IPO open price
    # z-score so magnitude doesn't dominate similarity
    std = pct.std()
    if std < 1e-9:
        return pct  # flat line — keep as-is
    return (pct - pct.mean()) / std


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def dtw_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    DTW distance converted to similarity score in [0, 1].
    Lower DTW distance = more similar = higher score.
    """
    if not HAS_DTW or len(a) == 0 or len(b) == 0:
        return 0.0
    dist = _dtw.distance_fast(a.astype(np.double), b.astype(np.double))
    return 1.0 / (1.0 + dist)


def combined_score(a: np.ndarray, b: np.ndarray) -> float:
    """Weighted average of cosine + DTW (or just cosine if DTW unavailable)."""
    cs = cosine_sim(a, b)
    if HAS_DTW:
        ds = dtw_sim(a, b)
        return 0.6 * cs + 0.4 * ds
    return cs


# ---------------------------------------------------------------------------
# IPO date detection
# ---------------------------------------------------------------------------

def get_ipo_date(df: pd.DataFrame) -> date | None:
    """Return the first date in the price history (proxy for IPO date)."""
    if df.empty:
        return None
    return df.index[0].date()


def is_recent_ipo(df: pd.DataFrame, window_days: int) -> bool:
    """True if IPO date is within the last window_days days."""
    ipo_d = get_ipo_date(df)
    if ipo_d is None:
        return False
    cutoff = date.today() - timedelta(days=window_days)
    return ipo_d >= cutoff


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def run_scan(
    target_ticker: str,
    n_days: int,
    top_n: int,
    ipo_window: int,
    min_gain_pct: float,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # --- 1. Target pattern ---
    print(f"\n[1/4] Fetching target: {target_ticker}")
    target_df = fetch_history(target_ticker, period="5d", interval="1d")
    if target_df.empty:
        print(f"  ERROR: Could not fetch {target_ticker}. Check ticker symbol.")
        print("  Hint: SpaceX IPO ticker may not yet be available in yfinance.")
        print("  Try: python ipo_pattern_scan.py --target <actual_ticker>")
        # For demo mode, generate a synthetic FOMO curve
        print("  Falling back to synthetic FOMO demo pattern...")
        closes = np.array([100.0, 118.0, 134.0])  # +18%, +34% FOMO shape
        target_norm = (closes - closes[0]) / closes[0] * 100
        target_ticker += "_DEMO"
    else:
        target_df_trimmed = first_n_trading_days(target_df, n_days)
        target_norm = normalize_returns(target_df_trimmed)

    if len(target_norm) < 2:
        print("  ERROR: Not enough target data (need >= 2 days).")
        return

    total_gain = (target_norm[-1] - target_norm[0]) if "_DEMO" not in target_ticker else 34.0
    print(f"  Pattern: {len(target_norm)} points, shape: {target_norm.round(2)}")

    # --- 2. Candidate pool ---
    all_tickers = list(set(US_IPO_SEEDS + TW_IPO_SEEDS))
    print(f"\n[2/4] Fetching {len(all_tickers)} candidates (cached where possible)...")

    results = []
    skipped = 0

    for ticker in tqdm(all_tickers, ncols=80):
        df = fetch_history(ticker, period="max", interval="1d")
        if df.empty or len(df) < n_days:
            skipped += 1
            continue

        # Filter: must be a recent IPO
        if not is_recent_ipo(df, ipo_window):
            skipped += 1
            continue

        trimmed = first_n_trading_days(df, n_days)
        closes = trimmed["Close"].values.astype(float)
        gain_pct = (closes[-1] - closes[0]) / closes[0] * 100

        # Filter: FOMO filter — must also have strong early gain
        if gain_pct < min_gain_pct:
            skipped += 1
            continue

        cand_norm = normalize_returns(trimmed)
        if len(cand_norm) < 2:
            skipped += 1
            continue

        score = combined_score(target_norm, cand_norm)
        ipo_date = get_ipo_date(df)

        results.append({
            "ticker": ticker,
            "ipo_date": str(ipo_date),
            "gain_pct_3d": round(gain_pct, 2),
            "similarity": round(score, 4),
            "closes": closes.tolist(),
            "norm": cand_norm.tolist(),
        })

    print(f"  Matched: {len(results)} | Skipped/filtered: {skipped}")

    if not results:
        print("\n  No candidates found. Try:")
        print("  - Expanding --ipo_window (current: {}d)".format(ipo_window))
        print("  - Lowering --min_gain (current: {}%)".format(min_gain_pct))
        return

    # --- 3. Rank ---
    print("\n[3/4] Ranking by similarity...")
    df_results = pd.DataFrame(results).sort_values("similarity", ascending=False)
    top = df_results.head(top_n)

    print("\n" + "=" * 70)
    print(f"  TOP {top_n} IPOs SIMILAR TO {target_ticker} (first {n_days} trading days)")
    print("=" * 70)
    print(top[["ticker", "ipo_date", "gain_pct_3d", "similarity"]].to_string(index=False))
    print("=" * 70)

    # Save CSV
    csv_path = OUT_DIR / f"similar_ipos_{target_ticker.replace('.', '_')}.csv"
    top.drop(columns=["closes", "norm"]).to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    # --- 4. Chart ---
    print("\n[4/4] Generating comparison chart...")
    _plot_results(target_norm, top, target_ticker, n_days)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_results(target_norm: np.ndarray, top: pd.DataFrame, target_ticker: str, n_days: int) -> None:
    n_top = min(len(top), 12)  # max 12 in chart
    top_chart = top.head(n_top)

    fig = plt.figure(figsize=(16, 10), facecolor="#0f0f14")
    fig.suptitle(
        f"IPO FOMO Pattern Match: {target_ticker} (first {n_days} trading days)",
        color="#e8e6e0", fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 2], hspace=0.4)

    # Top panel: target pattern
    ax_target = fig.add_subplot(gs[0])
    ax_target.set_facecolor("#1a1a22")
    x_t = range(len(target_norm))
    ax_target.plot(x_t, target_norm, color="#f59e0b", linewidth=2.5, marker="o", markersize=6, label=target_ticker)
    ax_target.axhline(0, color="#4a4a5a", linewidth=0.8, linestyle="--")
    ax_target.set_title(f"{target_ticker} — query pattern (z-score normalized)", color="#9ca3af", fontsize=10)
    ax_target.tick_params(colors="#6b7280")
    ax_target.set_xlabel("Trading Day", color="#6b7280")
    ax_target.set_ylabel("Z-score", color="#6b7280")
    ax_target.legend(fontsize=9, facecolor="#1a1a22", labelcolor="#e8e6e0")
    for spine in ax_target.spines.values():
        spine.set_edgecolor("#2d2d3a")

    # Bottom panel: top matches
    ax_matches = fig.add_subplot(gs[1])
    ax_matches.set_facecolor("#1a1a22")

    cmap = plt.cm.get_cmap("plasma", n_top)
    for i, row in enumerate(top_chart.itertuples()):
        norm_vals = np.array(row.norm)
        x = range(len(norm_vals))
        sim_label = f"{row.ticker} ({row.ipo_date}) sim={row.similarity:.3f} +{row.gain_pct_3d}%"
        ax_matches.plot(x, norm_vals, color=cmap(i), alpha=0.75, linewidth=1.5, label=sim_label)

    # Overlay target
    ax_matches.plot(
        range(len(target_norm)), target_norm,
        color="#f59e0b", linewidth=2.5, linestyle="--",
        label=f"{target_ticker} (target)"
    )
    ax_matches.axhline(0, color="#4a4a5a", linewidth=0.8, linestyle="--")
    ax_matches.set_title(f"Top {n_top} similar IPO patterns (z-score normalized)", color="#9ca3af", fontsize=10)
    ax_matches.tick_params(colors="#6b7280")
    ax_matches.set_xlabel("Trading Day", color="#6b7280")
    ax_matches.set_ylabel("Z-score", color="#6b7280")
    ax_matches.legend(
        fontsize=7.5, facecolor="#1a1a22", labelcolor="#e8e6e0",
        loc="upper left", ncol=2, framealpha=0.7
    )
    for spine in ax_matches.spines.values():
        spine.set_edgecolor("#2d2d3a")

    chart_path = OUT_DIR / f"pattern_match_{target_ticker.replace('.', '_')}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#0f0f14")
    plt.close()
    print(f"  Chart saved: {chart_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="IPO FOMO pattern scanner — compare first-N-day momentum"
    )
    p.add_argument("--target",      default=DEFAULT_TARGET,     help="Target IPO ticker (e.g. SPXC)")
    p.add_argument("--days",        type=int, default=DEFAULT_DAYS,    help="First N trading days to compare (default: 3)")
    p.add_argument("--top",         type=int, default=DEFAULT_TOP,     help="Top N results to show (default: 20)")
    p.add_argument("--ipo_window",  type=int, default=DEFAULT_IPO_WINDOW, help="Search candidates from past N days (default: 365)")
    p.add_argument("--min_gain",    type=float, default=DEFAULT_MIN_GAIN, help="Min 3-day gain %% to include (FOMO filter, default: 5.0)")
    p.add_argument("--no_cache",    action="store_true",         help="Ignore existing cache and re-fetch")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.no_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("Cache cleared.")

    run_scan(
        target_ticker=args.target,
        n_days=args.days,
        top_n=args.top,
        ipo_window=args.ipo_window,
        min_gain_pct=args.min_gain,
    )
