"""
ipo_pattern_scan.py  v6
-----------------------
Compares the first-N-day price momentum of a target IPO (e.g. SpaceX)
against all recent IPOs in US and TW markets, ranked by pattern similarity.

Fixes vs v5:
  - matplotlib get_cmap() deprecation fixed (use matplotlib.colormaps[])
  - skip report now shows low_gain tickers so you know what was filtered
  - minor: slice_ipo falls back to head(n) when ipo_date precedes history

Usage:
  python ipo_pattern_scan.py --target SPXC --days 3 --top 20
  python ipo_pattern_scan.py --target RDDT --min_gain 10 --ipo_window 1825
  python ipo_pattern_scan.py --target SPXC --no_cache

Dependencies:
  pip install yfinance pandas numpy matplotlib tqdm requests beautifulsoup4
  Optional: pip install dtaidistance
"""

import argparse
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib
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
    warnings.warn("requests/bs4 not installed; live IPO list disabled.")

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
DEFAULT_IPO_WINDOW = 1825   # 5 years — ensures seed list always has candidates
DEFAULT_MIN_GAIN   = 0.0    # 0 = no filter; set --min_gain 5 to focus on FOMO patterns
CACHE_DIR          = Path("ipo_scan_cache")
OUT_DIR            = Path("ipo_scan_results")
HDRS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
}


# ---------------------------------------------------------------------------
# Hardcoded seed list  (ticker -> IPO date YYYY-MM-DD)
# Completely offline-capable. Update this dict as new IPOs list.
# DEFAULT_IPO_WINDOW = 1825 days (5y); all entries below are within range.
# ---------------------------------------------------------------------------
SEED_WITH_DATES: dict[str, str] = {
    # ---- US 2021 ----
    "RKLB":  "2021-08-25",   # Rocket Lab (SPAC merge)
    "ASTS":  "2021-04-07",   # AST SpaceMobile
    "IONQ":  "2021-10-01",   # IonQ
    "ONON":  "2021-09-15",   # On Running
    "RIVN":  "2021-11-10",   # Rivian
    "DNUT":  "2021-07-01",   # Krispy Kreme
    "DUOL":  "2021-07-28",   # Duolingo
    "COIN":  "2021-04-14",   # Coinbase (direct listing)
    "AFRM":  "2021-01-13",   # Affirm
    "RBLX":  "2021-03-10",   # Roblox (direct listing)
    "HOOD":  "2021-07-29",   # Robinhood
    "MNDY":  "2021-06-11",   # Monday.com
    "GTLB":  "2021-10-14",   # GitLab
    "HIMS":  "2021-01-20",   # Hims & Hers
    "APP":   "2021-09-01",   # AppLovin
    "OPEN":  "2021-09-29",   # Opendoor Technologies
    "EXFY":  "2021-06-17",   # Expensify
    # ---- US 2022 ----
    "CRDO":  "2022-01-27",   # Credo Technology
    "DAVE":  "2022-01-05",   # Dave Inc (SPAC)
    "DBRG":  "2022-07-25",   # DigitalBridge
    # ---- US 2023 ----
    "ARM":   "2023-09-14",   # Arm Holdings
    "KVYO":  "2023-09-20",   # Klaviyo
    "CART":  "2023-09-19",   # Instacart
    "BIRK":  "2023-10-11",   # Birkenstock
    "CAVA":  "2023-06-15",   # Cava Group
    "LUNR":  "2023-12-15",   # Intuitive Machines
    # ---- US 2024 ----
    "RDDT":  "2024-03-21",   # Reddit
    "ACHR":  "2024-08-08",   # Archer Aviation
    "SEZL":  "2024-07-25",   # Sezzle
    "LOAR":  "2024-10-22",   # Loar Holdings
    "VERX":  "2024-03-28",   # Vertex Inc
    "STLC":  "2024-09-25",   # Stelco Holdings
    # ---- US 2025 ----
    "MNSB":  "2025-01-16",
    "CWAN":  "2025-03-19",
    "SFIN":  "2025-04-10",
    "CLPT":  "2025-05-08",
    # ---- TW 2021-2024 ----
    "6670.TW": "2021-09-15",
    "6756.TW": "2021-10-27",
    "6732.TW": "2021-05-18",
    "6789.TW": "2022-05-16",
    "6781.TW": "2022-03-29",
    "6768.TW": "2022-10-07",
    "6830.TW": "2023-02-14",
    "6916.TW": "2024-02-01",
    "6988.TW": "2025-05-20",
    "3680.TW": "2025-02-18",
    "6977.TW": "2025-04-22",
}


# ---------------------------------------------------------------------------
# Live IPO list — US: stockanalysis JSON API, fallback HTML scrape
# ---------------------------------------------------------------------------

def _sa_json(year: int, cutoff: date) -> list[str]:
    url = f"https://stockanalysis.com/api/ipos/?year={year}"
    try:
        r = requests.get(url, headers=HDRS, timeout=12)
        r.raise_for_status()
        data = r.json()
        out = []
        for item in data:
            sym = item.get("s") or item.get("symbol") or ""
            dt_str = item.get("ipoDate") or item.get("date") or ""
            if not sym or not dt_str:
                continue
            try:
                d = datetime.strptime(dt_str[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            if d >= cutoff:
                out.append(sym.upper())
        return out
    except Exception:
        return []


def _sa_html(year: int, cutoff: date) -> list[str]:
    url = f"https://stockanalysis.com/ipos/{year}/"
    try:
        r = requests.get(url, headers=HDRS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            return []
        out = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue
            sym = cols[0].get_text(strip=True).upper()
            dt_str = cols[2].get_text(strip=True)
            try:
                d = datetime.strptime(dt_str, "%b %d, %Y").date()
            except ValueError:
                continue
            if d >= cutoff:
                out.append(sym)
        return out
    except Exception:
        return []


def fetch_us_ipos(window_days: int) -> list[str]:
    if not HAS_REQUESTS:
        return []
    cutoff = date.today() - timedelta(days=window_days)
    tickers: list[str] = []
    years = sorted({date.today().year, date.today().year - 1,
                    date.today().year - 2}, reverse=True)
    for year in years:
        tickers += _sa_json(year, cutoff) or _sa_html(year, cutoff)
    print(f"  [US IPOs] {len(tickers)} tickers from stockanalysis")
    return tickers


def fetch_tw_ipos(window_days: int) -> list[str]:
    if not HAS_REQUESTS:
        return []
    cutoff = date.today() - timedelta(days=window_days)
    url = "https://openapi.twse.com.tw/v1/company/newlyListedStockInfo"
    tickers: list[str] = []
    try:
        r = requests.get(url, headers=HDRS, timeout=15)
        r.raise_for_status()
        data = r.json()
        for item in data:
            code = (
                item.get("SecuritiesCompanyCode")
                or item.get("stockCode")
                or item.get("Code")
                or ""
            ).strip()
            dt_str = (
                item.get("listingDate")
                or item.get("MarketEntryDate")
                or item.get("ListingDate")
                or ""
            ).strip()
            if not code or not dt_str:
                continue
            for fmt in ("%Y%m%d", "%Y/%m/%d", "%Y-%m-%d"):
                try:
                    d = datetime.strptime(dt_str, fmt).date()
                    break
                except ValueError:
                    d = None
            if d and d >= cutoff:
                tickers.append(f"{code}.TW")
    except Exception as e:
        warnings.warn(f"TWSE fetch failed: {e}")
    print(f"  [TW IPOs] {len(tickers)} tickers from TWSE")
    return tickers


# ---------------------------------------------------------------------------
# IPO date detection — 3-tier fallback
# ---------------------------------------------------------------------------

def _epoch_to_date(epoch) -> date | None:
    try:
        if epoch and int(epoch) > 0:
            return datetime.fromtimestamp(int(epoch), tz=timezone.utc).date()
    except Exception:
        pass
    return None


def get_ipo_date(ticker: str, ipo_window: int = DEFAULT_IPO_WINDOW) -> date | None:
    CACHE_DIR.mkdir(exist_ok=True)
    meta_path = CACHE_DIR / "ipo_dates.csv"

    if meta_path.exists():
        try:
            cache_df = pd.read_csv(meta_path, index_col="ticker", dtype={"ipo_date": "object"})
        except Exception:
            cache_df = pd.DataFrame({"ipo_date": pd.Series(dtype="object")})
            cache_df.index.name = "ticker"
    else:
        cache_df = pd.DataFrame({"ipo_date": pd.Series(dtype="object")})
        cache_df.index.name = "ticker"

    if ticker in cache_df.index:
        val = str(cache_df.loc[ticker, "ipo_date"])
        if val and val not in ("nan", "None", ""):
            try:
                return datetime.strptime(val, "%Y-%m-%d").date()
            except ValueError:
                pass

    ipo_d: date | None = None

    try:
        tk = yf.Ticker(ticker)

        # Tier 1: fast_info — try all known attribute names across yfinance versions
        fi = tk.fast_info
        for attr in ("first_trade_date_epoch_utc", "firstTradeDateEpochUtc",
                     "first_trade_date", "firstTradeDate"):
            val = getattr(fi, attr, None)
            if val is None:
                try:
                    val = fi[attr]
                except Exception:
                    pass
            ipo_d = _epoch_to_date(val)
            if ipo_d:
                break

        # Tier 2: full info dict (yfinance 0.2.x)
        if ipo_d is None:
            try:
                info = tk.info
                ipo_d = _epoch_to_date(info.get("firstTradeDateEpochUtc"))
            except Exception:
                pass

        # Tier 3: first row of history — only accepted if within ipo_window
        if ipo_d is None:
            try:
                hist = tk.history(period="max", interval="1d", auto_adjust=True)
                if not hist.empty:
                    first_day = pd.to_datetime(hist.index[0]).tz_localize(None).date()
                    cutoff_guard = date.today() - timedelta(days=ipo_window)
                    if first_day >= cutoff_guard:
                        ipo_d = first_day
            except Exception:
                pass

        time.sleep(0.25)

    except Exception as e:
        warnings.warn(f"[{ticker}] ipo_date lookup failed: {e}")

    cache_df.loc[ticker, "ipo_date"] = str(ipo_d) if ipo_d else ""
    cache_df.to_csv(meta_path)
    return ipo_d


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def fetch_history(ticker: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = CACHE_DIR / f"{ticker.replace('.', '_')}_1d.parquet"
    if cache_key.exists() and (time.time() - cache_key.stat().st_mtime) / 3600 < 4:
        try:
            return pd.read_parquet(cache_key)
        except Exception:
            pass
    try:
        df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.sort_index(inplace=True)
        df.to_parquet(cache_key)
        time.sleep(0.25)
        return df
    except Exception as e:
        warnings.warn(f"[{ticker}] history failed: {e}")
        return pd.DataFrame()


def slice_ipo(df: pd.DataFrame, ipo_d: date, n: int) -> pd.DataFrame:
    sliced = df[df.index >= pd.Timestamp(ipo_d)].head(n)
    if len(sliced) < 2:
        # ipo_d may be a weekend/holiday; fall back to first n rows
        sliced = df.head(n)
    return sliced


def normalize(closes: np.ndarray) -> np.ndarray:
    if len(closes) < 2 or closes[0] <= 0:
        return np.array([])
    pct = (closes - closes[0]) / closes[0] * 100.0
    std = pct.std()
    return pct if std < 1e-9 else (pct - pct.mean()) / std


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def dtw_sim(a: np.ndarray, b: np.ndarray) -> float:
    if not HAS_DTW or len(a) < 2 or len(b) < 2:
        return 0.0
    return 1.0 / (1.0 + _dtw.distance_fast(a.astype(np.double), b.astype(np.double)))


def score(a: np.ndarray, b: np.ndarray) -> float:
    cs = cosine_sim(a, b)
    return 0.6 * cs + 0.4 * dtw_sim(a, b) if HAS_DTW else cs


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def run_scan(target_ticker: str, n_days: int, top_n: int,
             ipo_window: int, min_gain_pct: float) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cutoff = date.today() - timedelta(days=ipo_window)

    # ---- 1. Target ----
    print(f"\n[1/4] Fetching target: {target_ticker}")
    demo_mode = False

    target_ipo = None
    if target_ticker in SEED_WITH_DATES:
        try:
            target_ipo = datetime.strptime(SEED_WITH_DATES[target_ticker], "%Y-%m-%d").date()
        except ValueError:
            pass
    if target_ipo is None:
        target_ipo = get_ipo_date(target_ticker, ipo_window=3650)

    target_df = fetch_history(target_ticker)

    if target_df.empty or target_ipo is None:
        print(f"  WARNING: {target_ticker} unavailable in yfinance.")
        print("  Using synthetic FOMO pattern (+10% / +22% / +30%)")
        raw_closes = np.array([100.0, 110.0, 122.0, 130.0])[:n_days]
        demo_mode = True
        target_ipo = date.today() - timedelta(days=3)
    else:
        sliced = slice_ipo(target_df, target_ipo, n_days)
        raw_closes = sliced["Close"].values.astype(float)

    target_norm = normalize(raw_closes)
    target_gain = (raw_closes[-1] - raw_closes[0]) / raw_closes[0] * 100 if not demo_mode else 30.0
    print(f"  IPO date  : {target_ipo}")
    print(f"  {n_days}-day gain: +{target_gain:.1f}%")
    print(f"  Pattern   : {target_norm.round(3)}")

    # ---- 2. Candidate pool ----
    print(f"\n[2/4] Building candidate list (window={ipo_window}d, cutoff={cutoff})...")
    live_us = fetch_us_ipos(ipo_window)
    live_tw = fetch_tw_ipos(ipo_window)

    seed_in_window = [
        t for t, ds in SEED_WITH_DATES.items()
        if datetime.strptime(ds, "%Y-%m-%d").date() >= cutoff
    ]
    print(f"  [Seeds]   {len(seed_in_window)} hardcoded tickers within window")

    all_candidates = list(
        set(live_us + live_tw + seed_in_window) - {target_ticker}
    )
    print(f"  Total     : {len(all_candidates)} unique candidates")

    if not all_candidates:
        print("\n  No candidates found from any source.")
        print(f"  Earliest seed = {min(SEED_WITH_DATES.values())}, cutoff = {cutoff}")
        print("  Try: --ipo_window 3650")
        return

    # ---- 3. Score ----
    print("\n[3/4] Scoring...")
    results = []
    s_no_date, s_old, s_gain, s_data = 0, 0, 0, 0
    low_gain_tickers: list[str] = []

    for ticker in tqdm(all_candidates, ncols=80):
        if ticker in SEED_WITH_DATES:
            try:
                ipo_d = datetime.strptime(SEED_WITH_DATES[ticker], "%Y-%m-%d").date()
            except ValueError:
                ipo_d = None
        else:
            ipo_d = get_ipo_date(ticker, ipo_window=ipo_window)

        if ipo_d is None:
            s_no_date += 1; continue
        if ipo_d < cutoff:
            s_old += 1; continue

        df = fetch_history(ticker)
        if df.empty:
            s_data += 1; continue

        sliced = slice_ipo(df, ipo_d, n_days)
        if len(sliced) < 2:
            s_data += 1; continue

        closes = sliced["Close"].values.astype(float)
        gain = (closes[-1] - closes[0]) / closes[0] * 100
        if gain < min_gain_pct:
            s_gain += 1
            low_gain_tickers.append(f"{ticker}({gain:+.1f}%)")
            continue

        cand_norm = normalize(closes)
        if len(cand_norm) < 2:
            s_data += 1; continue

        results.append({
            "ticker":      ticker,
            "ipo_date":    str(ipo_d),
            "gain_pct_3d": round(gain, 2),
            "similarity":  round(score(target_norm, cand_norm), 4),
            "closes":      closes.tolist(),
            "norm":        cand_norm.tolist(),
        })

    print(f"  Matched : {len(results)}")
    print(f"  Skipped : no_date={s_no_date} too_old={s_old} "
          f"low_gain={s_gain} no_data={s_data}")
    if low_gain_tickers and min_gain_pct > 0:
        print(f"  low_gain tickers ({min_gain_pct:.0f}% threshold):")
        print(f"    {', '.join(low_gain_tickers)}")
        print(f"  Tip: run with --min_gain 0 to include all")

    if not results:
        print("\n  No matches found.  Suggestions:")
        print("    --min_gain 0           disable gain filter")
        print("    --ipo_window 3650      widen date window")
        print("    --days 2               reduce required trading days")
        return

    # ---- 4. Output ----
    print("\n[4/4] Ranking...")
    df_out = pd.DataFrame(results).sort_values("similarity", ascending=False)
    top = df_out.head(top_n)
    label = f"{target_ticker}{'_DEMO' if demo_mode else ''}"

    print("\n" + "=" * 72)
    print(f"  TOP {top_n} SIMILAR IPOs — {label}  (first {n_days} days)")
    print("=" * 72)
    print(top[["ticker", "ipo_date", "gain_pct_3d", "similarity"]].to_string(index=False))
    print("=" * 72)

    csv_path = OUT_DIR / f"similar_ipos_{label.replace('.','_')}.csv"
    top.drop(columns=["closes", "norm"]).to_csv(csv_path, index=False)
    print(f"  CSV   -> {csv_path}")
    _plot(target_norm, raw_closes, top, label, n_days)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(target_norm, raw_closes, top, label, n_days):
    n_show = min(len(top), 12)
    fig = plt.figure(figsize=(16, 11), facecolor="#0f0f14")
    fig.suptitle(f"IPO FOMO Pattern Match  ·  {label}  (first {n_days} days)",
                 color="#e8e6e0", fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.2], hspace=0.45)

    # fix: use matplotlib.colormaps instead of deprecated plt.cm.get_cmap
    cmap = matplotlib.colormaps.get_cmap("plasma").resampled(max(n_show, 1))

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#1a1a22")
    base = raw_closes[0]
    pct_raw = (raw_closes - base) / base * 100
    xs = list(range(1, len(pct_raw) + 1))
    ax1.bar(xs, pct_raw, color="#f59e0b", alpha=0.65, width=0.5)
    ax1.plot(xs, pct_raw, color="#f59e0b", lw=2, marker="o", ms=7)
    for i, v in enumerate(pct_raw):
        ax1.text(xs[i], v + 0.3, f"+{v:.1f}%", ha="center", va="bottom",
                 color="#fcd34d", fontsize=9, fontweight="bold")
    ax1.axhline(0, color="#4a4a5a", lw=0.8, ls="--")
    ax1.set_title(f"{label}  —  % gain from IPO open", color="#9ca3af", fontsize=10)
    ax1.set_xlabel("Trading Day", color="#6b7280", fontsize=9)
    ax1.set_ylabel("% from open", color="#6b7280", fontsize=9)
    ax1.tick_params(colors="#6b7280")
    for sp in ax1.spines.values(): sp.set_edgecolor("#2d2d3a")

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1a1a22")
    for i, row in enumerate(top.head(n_show).itertuples()):
        nv = np.array(row.norm)
        ax2.plot(range(1, len(nv)+1), nv, color=cmap(i / max(n_show - 1, 1)),
                 alpha=0.75, lw=1.5,
                 label=f"{row.ticker} ({row.ipo_date})  {row.similarity:.3f}  +{row.gain_pct_3d}%")
    ax2.plot(range(1, len(target_norm)+1), target_norm,
             color="#f59e0b", lw=2.8, ls="--", label=f"{label} (target)")
    ax2.axhline(0, color="#4a4a5a", lw=0.8, ls="--")
    ax2.set_title(f"Top {n_show} similar patterns (z-score normalized)", color="#9ca3af", fontsize=10)
    ax2.set_xlabel("Trading Day", color="#6b7280", fontsize=9)
    ax2.set_ylabel("Z-score", color="#6b7280", fontsize=9)
    ax2.tick_params(colors="#6b7280")
    ax2.legend(fontsize=7.5, facecolor="#1a1a22", labelcolor="#e8e6e0",
               loc="upper left", ncol=2, framealpha=0.75)
    for sp in ax2.spines.values(): sp.set_edgecolor("#2d2d3a")

    safe = label.replace(".", "_").replace("(", "").replace(")", "")
    path = OUT_DIR / f"pattern_match_{safe}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f0f14")
    plt.close()
    print(f"  Chart -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="IPO FOMO pattern scanner v6")
    p.add_argument("--target",     default=DEFAULT_TARGET)
    p.add_argument("--days",       type=int,   default=DEFAULT_DAYS)
    p.add_argument("--top",        type=int,   default=DEFAULT_TOP)
    p.add_argument("--ipo_window", type=int,   default=DEFAULT_IPO_WINDOW,
                   help="Days back to search for comparable IPOs (default 1825 = 5y)")
    p.add_argument("--min_gain",   type=float, default=DEFAULT_MIN_GAIN,
                   help="Min N-day gain %% to include candidate (default 0 = no filter)")
    p.add_argument("--no_cache",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_cache and CACHE_DIR.exists():
        import shutil; shutil.rmtree(CACHE_DIR)
        print("Cache cleared.")
    run_scan(
        target_ticker=args.target,
        n_days=args.days,
        top_n=args.top,
        ipo_window=args.ipo_window,
        min_gain_pct=args.min_gain,
    )
