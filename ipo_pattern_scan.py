"""
ipo_pattern_scan.py
-------------------
Compares the first-3-day price momentum of a target IPO (e.g. SpaceX)
against all recent IPOs in US and Taiwan markets, then ranks them by
pattern similarity.

Bug fixes vs v1:
  - IPO date now read from yf.Ticker.fast_info.first_trade_date_epoch_utc
    (accurate) instead of df.index[0] on period="max" (wrong for old stocks)
  - Candidate list fetched live from stockanalysis.com IPO screener
    instead of a static seed list full of delisted tickers
  - Taiwan IPOs fetched from TWSE open data API

Pipeline:
  1. Fetch target ticker's first N trading days
  2. Collect recent IPO tickers (US: stockanalysis scrape, TW: TWSE API)
  3. Verify each ticker's actual IPO date via fast_info
  4. Normalize series (z-score on % returns from IPO open)
  5. Score via cosine similarity + optional DTW
  6. Print ranked table + save chart to ipo_scan_results/

Usage:
  python ipo_pattern_scan.py --target SPXC --days 3 --top 20
  python ipo_pattern_scan.py --target SPXC --days 3 --ipo_window 730 --min_gain 3
  python ipo_pattern_scan.py --target SPXC --no_cache   # clear cache

Dependencies:
  pip install yfinance pandas numpy matplotlib tqdm requests beautifulsoup4
  Optional: pip install dtaidistance  # enables DTW scoring
"""

import argparse
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance not found.  pip install yfinance")

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    warnings.warn("requests/beautifulsoup4 not installed; live IPO list disabled.  pip install requests beautifulsoup4")

try:
    from dtaidistance import dtw as _dtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_TARGET     = "SPXC"
DEFAULT_DAYS       = 3
DEFAULT_TOP        = 20
DEFAULT_IPO_WINDOW = 365   # days back to search
DEFAULT_MIN_GAIN   = 5.0   # % gain in first N days (FOMO filter)
CACHE_DIR          = Path("ipo_scan_cache")
OUT_DIR            = Path("ipo_scan_results")
REQUEST_HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; ipo-scanner/2.0)"}


# ---------------------------------------------------------------------------
# Live IPO list fetching
# ---------------------------------------------------------------------------

def fetch_us_ipos_stockanalysis(window_days: int) -> list[str]:
    """
    Scrape recent US IPO tickers from stockanalysis.com/ipos/
    Returns list of ticker strings.
    Falls back to an empty list on any error.
    """
    if not HAS_REQUESTS:
        return []

    tickers = []
    cutoff = date.today() - timedelta(days=window_days)
    # stockanalysis lists IPOs by year; check current + previous year
    for year in sorted({date.today().year, date.today().year - 1}, reverse=True):
        url = f"https://stockanalysis.com/ipos/{year}/"
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                continue
            for row in table.find_all("tr")[1:]:  # skip header
                cols = row.find_all("td")
                if len(cols) < 3:
                    continue
                # cols: Symbol | Company | IPO Date | ...
                ticker = cols[0].get_text(strip=True)
                ipo_date_str = cols[2].get_text(strip=True)
                try:
                    ipo_d = datetime.strptime(ipo_date_str, "%b %d, %Y").date()
                except ValueError:
                    continue
                if ipo_d >= cutoff:
                    tickers.append(ticker)
        except Exception as e:
            warnings.warn(f"stockanalysis fetch failed ({url}): {e}")

    print(f"  [US IPOs] {len(tickers)} tickers from stockanalysis.com")
    return tickers


def fetch_tw_ipos_twse(window_days: int) -> list[str]:
    """
    Fetch recent Taiwan IPO (first listing) stocks from TWSE open data.
    Returns list of tickers in '<code>.TW' format.
    """
    if not HAS_REQUESTS:
        return []

    tickers = []
    cutoff = date.today() - timedelta(days=window_days)
    # TWSE open API: newly listed stocks
    url = "https://openapi.twse.com.tw/v1/company/newlyListedStockInfo"
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for item in data:
            code = item.get("SecuritiesCompanyCode", "").strip()
            listing_date_str = item.get("MarketEntryDate", "").strip()  # e.g. "20240315"
            if not code or not listing_date_str:
                continue
            try:
                listing_d = datetime.strptime(listing_date_str, "%Y%m%d").date()
            except ValueError:
                continue
            if listing_d >= cutoff:
                tickers.append(f"{code}.TW")
    except Exception as e:
        warnings.warn(f"TWSE API fetch failed: {e}")

    print(f"  [TW IPOs] {len(tickers)} tickers from TWSE open data")
    return tickers


# Fallback static list — only used when network unavailable
# Manually verified as of 2025-2026; smaller and cleaner than v1
US_FALLBACK = [
    "ARM", "KVYO", "CART", "BIRK", "CAVA", "RDDT",
    "RKLB", "ACHR", "ASTS", "LUNR", "IONQ", "SEZL",
    "LOAR", "VERX", "CRDO", "STLC", "ONON",
]
TW_FALLBACK = [
    "6669.TW", "6770.TW", "6515.TW", "3533.TW", "6789.TW",
    "6278.TW", "6768.TW", "6830.TW", "3711.TW", "2454.TW",
]


# ---------------------------------------------------------------------------
# IPO date: read from yfinance fast_info (the correct approach)
# ---------------------------------------------------------------------------

def get_ipo_date_from_yf(ticker: str) -> date | None:
    """
    Read firstTradeDateEpochUtc from yfinance fast_info.
    This is the actual first trading date regardless of how far back
    history() returns data — fixing the v1 bug.
    Cache result in a lightweight CSV to avoid repeated API calls.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    meta_cache = CACHE_DIR / "ipo_dates.csv"

    # Load existing cache
    if meta_cache.exists():
        try:
            cache_df = pd.read_csv(meta_cache, index_col="ticker")
        except Exception:
            cache_df = pd.DataFrame(columns=["ipo_date"])
            cache_df.index.name = "ticker"
    else:
        cache_df = pd.DataFrame(columns=["ipo_date"])
        cache_df.index.name = "ticker"

    if ticker in cache_df.index:
        val = cache_df.loc[ticker, "ipo_date"]
        if pd.notna(val):
            try:
                return datetime.strptime(str(val), "%Y-%m-%d").date()
            except ValueError:
                pass

    # Fetch from yfinance
    ipo_d = None
    try:
        fi = yf.Ticker(ticker).fast_info
        epoch = getattr(fi, "first_trade_date_epoch_utc", None)
        if epoch and epoch > 0:
            ipo_d = datetime.fromtimestamp(epoch, tz=timezone.utc).date()
        time.sleep(0.2)
    except Exception as e:
        warnings.warn(f"[{ticker}] fast_info failed: {e}")

    # Write back
    cache_df.loc[ticker, "ipo_date"] = str(ipo_d) if ipo_d else ""
    cache_df.to_csv(meta_cache)
    return ipo_d


# ---------------------------------------------------------------------------
# Price data helpers
# ---------------------------------------------------------------------------

def fetch_history_from_ipo(ticker: str, n_days: int) -> pd.DataFrame:
    """
    Fetch price history starting from the IPO date.
    Uses period="max" but slices from first_trade_date onward.
    Caches parquet per ticker.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = CACHE_DIR / f"{ticker.replace('.', '_')}_1d.parquet"

    if cache_key.exists():
        age_h = (time.time() - cache_key.stat().st_mtime) / 3600
        if age_h < 4:
            try:
                return pd.read_parquet(cache_key)
            except Exception:
                pass

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="max", interval="1d", auto_adjust=True)
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.sort_index(inplace=True)
        df.to_parquet(cache_key)
        time.sleep(0.3)
        return df
    except Exception as e:
        warnings.warn(f"[{ticker}] history fetch failed: {e}")
        return pd.DataFrame()


def slice_from_ipo(df: pd.DataFrame, ipo_d: date, n_days: int) -> pd.DataFrame:
    """
    Slice N rows starting from ipo_d.
    This correctly handles the case where history() includes pre-listing data.
    """
    ipo_ts = pd.Timestamp(ipo_d)
    sliced = df[df.index >= ipo_ts]
    return sliced.head(n_days)


def normalize_returns(closes: np.ndarray) -> np.ndarray:
    """z-score of % returns from base price."""
    if len(closes) < 2:
        return np.array([])
    base = closes[0]
    if base <= 0:
        return np.array([])
    pct = (closes - base) / base * 100.0
    std = pct.std()
    if std < 1e-9:
        return pct
    return (pct - pct.mean()) / std


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def dtw_sim(a: np.ndarray, b: np.ndarray) -> float:
    if not HAS_DTW or len(a) < 2 or len(b) < 2:
        return 0.0
    dist = _dtw.distance_fast(a.astype(np.double), b.astype(np.double))
    return 1.0 / (1.0 + dist)


def combined_score(a: np.ndarray, b: np.ndarray) -> float:
    cs = cosine_sim(a, b)
    if HAS_DTW:
        return 0.6 * cs + 0.4 * dtw_sim(a, b)
    return cs


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
    cutoff = date.today() - timedelta(days=ipo_window)

    # ---- 1. Target pattern ----
    print(f"\n[1/4] Fetching target: {target_ticker}")
    demo_mode = False
    target_ipo = get_ipo_date_from_yf(target_ticker)
    target_df = fetch_history_from_ipo(target_ticker, n_days)

    if target_df.empty or target_ipo is None:
        print(f"  WARNING: {target_ticker} not found in yfinance.")
        print("  Using synthetic FOMO demo pattern (+10% / +22% / +30%)")
        raw_closes = np.array([100.0, 110.0, 122.0, 130.0])[:n_days]
        demo_mode = True
        target_ipo = date.today() - timedelta(days=3)
    else:
        trimmed = slice_from_ipo(target_df, target_ipo, n_days)
        if len(trimmed) < 2:
            print(f"  WARNING: only {len(trimmed)} days available for {target_ticker}.")
            print("  Using all available data.")
            trimmed = target_df.head(n_days)
        raw_closes = trimmed["Close"].values.astype(float)

    target_norm = normalize_returns(raw_closes)
    target_gain = (raw_closes[-1] - raw_closes[0]) / raw_closes[0] * 100 if not demo_mode else 30.0
    print(f"  IPO date : {target_ipo}")
    print(f"  3-day gain: +{target_gain:.1f}%")
    print(f"  Pattern shape: {target_norm.round(3)}")

    # ---- 2. Candidate pool ----
    print(f"\n[2/4] Building candidate IPO list (window: last {ipo_window}d)...")
    us_tickers = fetch_us_ipos_stockanalysis(ipo_window)
    tw_tickers = fetch_tw_ipos_twse(ipo_window)

    if not us_tickers and not tw_tickers:
        print("  Network fetch failed; using fallback seed list.")
        us_tickers = US_FALLBACK
        tw_tickers = TW_FALLBACK

    # Remove target itself
    all_candidates = list(set(us_tickers + tw_tickers) - {target_ticker})
    print(f"  Total candidates: {len(all_candidates)}")

    # ---- 3. Score each candidate ----
    print(f"\n[3/4] Scoring candidates...")
    results = []
    skipped_no_ipo = 0
    skipped_old = 0
    skipped_gain = 0
    skipped_data = 0

    for ticker in tqdm(all_candidates, ncols=80):
        # Step A: verify IPO date via fast_info (the fix)
        ipo_d = get_ipo_date_from_yf(ticker)
        if ipo_d is None:
            skipped_no_ipo += 1
            continue
        if ipo_d < cutoff:
            skipped_old += 1
            continue

        # Step B: fetch and slice from actual IPO date
        df = fetch_history_from_ipo(ticker, n_days)
        if df.empty:
            skipped_data += 1
            continue

        sliced = slice_from_ipo(df, ipo_d, n_days)
        if len(sliced) < 2:
            skipped_data += 1
            continue

        closes = sliced["Close"].values.astype(float)
        gain_pct = (closes[-1] - closes[0]) / closes[0] * 100

        if gain_pct < min_gain_pct:
            skipped_gain += 1
            continue

        cand_norm = normalize_returns(closes)
        if len(cand_norm) < 2:
            skipped_data += 1
            continue

        score = combined_score(target_norm, cand_norm)
        results.append({
            "ticker":      ticker,
            "ipo_date":    str(ipo_d),
            "gain_pct_3d": round(gain_pct, 2),
            "similarity":  round(score, 4),
            "closes":      closes.tolist(),
            "norm":        cand_norm.tolist(),
        })

    print(f"  Matched : {len(results)}")
    print(f"  Skipped : no IPO date={skipped_no_ipo}, too old={skipped_old}, "
          f"low gain={skipped_gain}, no data={skipped_data}")

    if not results:
        print("\n  No candidates passed all filters. Suggestions:")
        print(f"  - Lower --min_gain (current: {min_gain_pct}%)")
        print(f"  - Expand --ipo_window (current: {ipo_window}d)")
        print("  - Check internet connection for live IPO list fetch")
        return

    # ---- 4. Rank + output ----
    print("\n[4/4] Ranking and plotting...")
    df_out = pd.DataFrame(results).sort_values("similarity", ascending=False)
    top = df_out.head(top_n)

    label = f"{target_ticker}{'(DEMO)' if demo_mode else ''}"
    print("\n" + "=" * 72)
    print(f"  TOP {top_n} IPOs SIMILAR TO {label}  (first {n_days} trading days)")
    print("=" * 72)
    print(top[["ticker", "ipo_date", "gain_pct_3d", "similarity"]].to_string(index=False))
    print("=" * 72)

    csv_path = OUT_DIR / f"similar_ipos_{target_ticker.replace('.', '_')}.csv"
    top.drop(columns=["closes", "norm"]).to_csv(csv_path, index=False)
    print(f"  CSV  -> {csv_path}")

    _plot_results(target_norm, raw_closes, top, label, n_days)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_results(
    target_norm: np.ndarray,
    raw_closes: np.ndarray,
    top: pd.DataFrame,
    label: str,
    n_days: int,
) -> None:
    n_show = min(len(top), 12)
    top_chart = top.head(n_show)

    fig = plt.figure(figsize=(16, 11), facecolor="#0f0f14")
    fig.suptitle(
        f"IPO FOMO Pattern Match  ·  {label}  (first {n_days} trading days)",
        color="#e8e6e0", fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.2], hspace=0.45)

    # --- top panel: target raw % returns ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#1a1a22")
    base = raw_closes[0]
    pct_raw = (raw_closes - base) / base * 100
    days_x = list(range(1, len(pct_raw) + 1))
    ax1.bar(days_x, pct_raw, color="#f59e0b", alpha=0.7, width=0.5)
    ax1.plot(days_x, pct_raw, color="#f59e0b", linewidth=2, marker="o", markersize=7)
    for i, v in enumerate(pct_raw):
        ax1.text(days_x[i], v + 0.3, f"+{v:.1f}%", ha="center", va="bottom",
                 color="#fcd34d", fontsize=9, fontweight="bold")
    ax1.axhline(0, color="#4a4a5a", linewidth=0.8, linestyle="--")
    ax1.set_title(f"{label}  —  raw % gain from IPO open", color="#9ca3af", fontsize=10)
    ax1.set_xlabel("Trading Day", color="#6b7280", fontsize=9)
    ax1.set_ylabel("% from open", color="#6b7280", fontsize=9)
    ax1.tick_params(colors="#6b7280")
    for sp in ax1.spines.values(): sp.set_edgecolor("#2d2d3a")

    # --- bottom panel: normalized shape comparison ---
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1a1a22")
    cmap = plt.cm.get_cmap("plasma", n_show)
    for i, row in enumerate(top_chart.itertuples()):
        nv = np.array(row.norm)
        sim_lbl = f"{row.ticker} ({row.ipo_date})  sim={row.similarity:.3f}  +{row.gain_pct_3d}%"
        ax2.plot(range(1, len(nv)+1), nv, color=cmap(i), alpha=0.75,
                 linewidth=1.5, label=sim_lbl)
    ax2.plot(range(1, len(target_norm)+1), target_norm,
             color="#f59e0b", linewidth=2.8, linestyle="--",
             label=f"{label} (target)")
    ax2.axhline(0, color="#4a4a5a", linewidth=0.8, linestyle="--")
    ax2.set_title(f"Top {n_show} similar IPO momentum patterns (z-score normalized)",
                  color="#9ca3af", fontsize=10)
    ax2.set_xlabel("Trading Day", color="#6b7280", fontsize=9)
    ax2.set_ylabel("Z-score", color="#6b7280", fontsize=9)
    ax2.tick_params(colors="#6b7280")
    ax2.legend(fontsize=7.5, facecolor="#1a1a22", labelcolor="#e8e6e0",
               loc="upper left", ncol=2, framealpha=0.75)
    for sp in ax2.spines.values(): sp.set_edgecolor("#2d2d3a")

    chart_path = OUT_DIR / f"pattern_match_{label.replace('.', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#0f0f14")
    plt.close()
    print(f"  Chart -> {chart_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="IPO FOMO pattern scanner")
    p.add_argument("--target",     default=DEFAULT_TARGET,      help="Target IPO ticker")
    p.add_argument("--days",       type=int, default=DEFAULT_DAYS,     help="First N trading days (default: 3)")
    p.add_argument("--top",        type=int, default=DEFAULT_TOP,      help="Top N to display (default: 20)")
    p.add_argument("--ipo_window", type=int, default=DEFAULT_IPO_WINDOW, help="Search window in days (default: 365)")
    p.add_argument("--min_gain",   type=float, default=DEFAULT_MIN_GAIN, help="Min gain %% filter (default: 5.0)")
    p.add_argument("--no_cache",   action="store_true",          help="Clear cache before running")
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
