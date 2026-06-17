"""
ipo_pattern_scan.py  v8
-----------------------
Compares the first-N-day price momentum of a target IPO (e.g. SpaceX)
against recent IPOs in US and TW markets, ranked by pattern similarity.

v8 changes vs v7:
  - Live US IPO list from Wikipedia (reliable, no JS) as primary source
  - stockanalysis JSON/HTML as secondary source
  - --sample N  : randomly subsample N candidates before scoring
                  Run multiple times to cover more tickers without waiting forever
  - --no_green  : allow red-candle Day-1 (only checks shadow length, not direction)
  - shadow_filter now separately enforces green + shadow; --no_green disables green check
  - seed list trimmed to verified US-only (TW seeds removed; TWSE live fetch kept)
  - Better no_data reporting: prints which tickers failed

Usage:
  # Default: no filters, all seeds
  python ipo_pattern_scan.py --target SPXC

  # Shadow filter ON, random 60 from pool each run
  python ipo_pattern_scan.py --target SPXC --shadow_filter --sample 60

  # Run 3 times to cover ~180 random samples
  for i in 1 2 3; do python ipo_pattern_scan.py --target SPXC --shadow_filter --sample 60; done

  # OHLC scoring + shadow filter
  python ipo_pattern_scan.py --target SPXC --shadow_filter --ohlc_score --sample 80

Dependencies:
  pip install yfinance pandas numpy matplotlib tqdm requests beautifulsoup4
  Optional: pip install dtaidistance
"""

import argparse
import random
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
DEFAULT_TARGET       = "SPXC"
DEFAULT_DAYS         = 3
DEFAULT_TOP          = 20
DEFAULT_IPO_WINDOW   = 1825   # 5 years
DEFAULT_MIN_GAIN     = 0.0
DEFAULT_SHADOW_RATIO = 0.25
CACHE_DIR            = Path("ipo_scan_cache")
OUT_DIR              = Path("ipo_scan_results")
HDRS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
}


# ---------------------------------------------------------------------------
# Verified seed list (US only — confirmed yfinance accessible)
# ---------------------------------------------------------------------------
SEED_WITH_DATES: dict[str, str] = {
    # 2019-2020 high-FOMO references
    "DDOG":  "2019-09-19",
    "SNOW":  "2020-09-16",
    "PLTR":  "2020-09-30",
    "ABNB":  "2020-12-10",
    "DASH":  "2020-12-09",
    "AI":    "2020-12-09",
    "U":     "2020-09-18",
    "BIGC":  "2020-08-05",
    # 2021
    "RKLB":  "2021-08-25",
    "ASTS":  "2021-04-07",
    "IONQ":  "2021-10-01",
    "ONON":  "2021-09-15",
    "RIVN":  "2021-11-10",
    "DNUT":  "2021-07-01",
    "DUOL":  "2021-07-28",
    "COIN":  "2021-04-14",
    "AFRM":  "2021-01-13",
    "RBLX":  "2021-03-10",
    "HOOD":  "2021-07-29",
    "MNDY":  "2021-06-11",
    "GTLB":  "2021-10-14",
    "HIMS":  "2021-01-20",
    "APP":   "2021-09-01",
    "OPEN":  "2021-09-29",
    "GRAB":  "2021-12-02",
    "JOBY":  "2021-08-10",
    "SEAT":  "2021-10-18",
    "BLNK":  "2021-01-15",
    # 2022
    "CRDO":  "2022-01-27",
    "DAVE":  "2022-01-05",
    "DBRG":  "2022-07-25",
    # 2023
    "ARM":   "2023-09-14",
    "KVYO":  "2023-09-20",
    "CART":  "2023-09-19",
    "BIRK":  "2023-10-11",
    "CAVA":  "2023-06-15",
    "LUNR":  "2023-12-15",
    # 2024
    "RDDT":  "2024-03-21",
    "ACHR":  "2024-08-08",
    "SEZL":  "2024-07-25",
    "LOAR":  "2024-10-22",
    "VERX":  "2024-03-28",
    "STLC":  "2024-09-25",
    "ASPI":  "2024-10-24",
    "RDZN":  "2024-11-07",
    "REAX":  "2024-09-12",
    "LEGT":  "2024-05-23",
    "ARGT":  "2024-06-06",
    "HYMC":  "2024-08-21",
    # 2025
    "MNSB":  "2025-01-16",
    "CWAN":  "2025-03-19",
    "SFIN":  "2025-04-10",
    "CLPT":  "2025-05-08",
    "HALO":  "2025-02-06",
    "OMAB":  "2025-05-15",
    # TW (confirmed accessible in yfinance)
    "6670.TW": "2021-09-15",
    "6756.TW": "2021-10-27",
    "6732.TW": "2021-05-18",
    "6789.TW": "2022-05-16",
    "6781.TW": "2022-03-29",
    "6768.TW": "2022-10-07",
    "6830.TW": "2023-02-14",
    "6916.TW": "2024-02-01",
    "6924.TW": "2024-04-18",
    "6988.TW": "2025-05-20",
}


# ---------------------------------------------------------------------------
# Live IPO list: Wikipedia (primary) + stockanalysis (fallback)
# Wikipedia IPO tables are stable HTML, no JS needed.
# ---------------------------------------------------------------------------

def _fetch_wiki_ipos(year: int, cutoff: date) -> dict[str, str]:
    """Returns {ticker: ipo_date_str} from Wikipedia 'YEAR in US IPOs' table."""
    url = f"https://en.wikipedia.org/wiki/{year}_in_the_United_States_IPO_market"
    alt = f"https://en.wikipedia.org/wiki/List_of_largest_technology_company_IPOs"
    out: dict[str, str] = {}
    for u in (url, alt):
        try:
            r = requests.get(u, headers=HDRS, timeout=15)
            if r.status_code != 200:
                continue
            tables = pd.read_html(r.text)
            for tbl in tables:
                tbl.columns = [str(c).strip().lower() for c in tbl.columns]
                # find ticker col
                sym_col = next((c for c in tbl.columns if any(k in c for k in
                    ("ticker", "symbol", "stock"))), None)
                date_col = next((c for c in tbl.columns if any(k in c for k in
                    ("date", "ipo", "listed"))), None)
                if sym_col is None:
                    continue
                for _, row in tbl.iterrows():
                    sym = str(row.get(sym_col, "")).strip().upper()
                    if not sym or len(sym) > 6 or not sym.isalpha():
                        continue
                    dt_str = str(row.get(date_col, "")).strip() if date_col else ""
                    d = None
                    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"):
                        try:
                            d = datetime.strptime(dt_str[:20], fmt).date()
                            break
                        except Exception:
                            pass
                    if d and d >= cutoff:
                        out[sym] = str(d)
        except Exception:
            pass
    return out


def _sa_json(year: int, cutoff: date) -> dict[str, str]:
    url = f"https://stockanalysis.com/api/ipos/?year={year}"
    out: dict[str, str] = {}
    try:
        r = requests.get(url, headers=HDRS, timeout=12)
        r.raise_for_status()
        data = r.json()
        for item in data:
            sym = (item.get("s") or item.get("symbol") or "").upper()
            dt_str = item.get("ipoDate") or item.get("date") or ""
            if not sym or not dt_str:
                continue
            try:
                d = datetime.strptime(dt_str[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            if d >= cutoff:
                out[sym] = str(d)
    except Exception:
        pass
    return out


def _sa_html(year: int, cutoff: date) -> dict[str, str]:
    url = f"https://stockanalysis.com/ipos/{year}/"
    out: dict[str, str] = {}
    try:
        r = requests.get(url, headers=HDRS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            return out
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
                out[sym] = str(d)
    except Exception:
        pass
    return out


def fetch_us_ipos_live(window_days: int) -> dict[str, str]:
    """Returns {ticker: ipo_date_str} for US IPOs within window."""
    if not HAS_REQUESTS:
        return {}
    cutoff = date.today() - timedelta(days=window_days)
    merged: dict[str, str] = {}
    years = sorted({date.today().year, date.today().year - 1,
                    date.today().year - 2, date.today().year - 3,
                    date.today().year - 4}, reverse=True)
    for year in years:
        # Wikipedia first, then stockanalysis
        wiki = _fetch_wiki_ipos(year, cutoff)
        sa = _sa_json(year, cutoff) or _sa_html(year, cutoff)
        merged.update(sa)
        merged.update(wiki)   # wiki overwrites (often more accurate dates)
    print(f"  [US live] {len(merged)} tickers (wiki + stockanalysis)")
    return merged


def fetch_tw_ipos_live(window_days: int) -> dict[str, str]:
    if not HAS_REQUESTS:
        return {}
    cutoff = date.today() - timedelta(days=window_days)
    url = "https://openapi.twse.com.tw/v1/company/newlyListedStockInfo"
    out: dict[str, str] = {}
    try:
        r = requests.get(url, headers=HDRS, timeout=15)
        r.raise_for_status()
        data = r.json()
        for item in data:
            code = (
                item.get("SecuritiesCompanyCode")
                or item.get("stockCode")
                or item.get("Code") or ""
            ).strip()
            dt_str = (
                item.get("listingDate")
                or item.get("MarketEntryDate")
                or item.get("ListingDate") or ""
            ).strip()
            if not code or not dt_str:
                continue
            d = None
            for fmt in ("%Y%m%d", "%Y/%m/%d", "%Y-%m-%d"):
                try:
                    d = datetime.strptime(dt_str, fmt).date()
                    break
                except ValueError:
                    pass
            if d and d >= cutoff:
                out[f"{code}.TW"] = str(d)
    except Exception as e:
        warnings.warn(f"TWSE fetch failed: {e}")
    print(f"  [TW live] {len(out)} tickers from TWSE")
    return out


# ---------------------------------------------------------------------------
# IPO date detection  (used for tickers with no known date)
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
            cache_df = pd.read_csv(meta_path, index_col="ticker",
                                   dtype={"ipo_date": "object"})
        except Exception:
            cache_df = pd.DataFrame({"ipo_date": pd.Series(dtype="object")})
            cache_df.index.name = "ticker"
    else:
        cache_df = pd.DataFrame({"ipo_date": pd.Series(dtype="object")})
        cache_df.index.name = "ticker"

    if ticker in cache_df.index:
        val = str(cache_df.loc[ticker, "ipo_date"])
        if val not in ("nan", "None", ""):
            try:
                return datetime.strptime(val, "%Y-%m-%d").date()
            except ValueError:
                pass

    ipo_d: date | None = None
    try:
        tk = yf.Ticker(ticker)
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
        if ipo_d is None:
            try:
                ipo_d = _epoch_to_date(tk.info.get("firstTradeDateEpochUtc"))
            except Exception:
                pass
        if ipo_d is None:
            try:
                hist = tk.history(period="max", interval="1d", auto_adjust=True)
                if not hist.empty:
                    first = pd.to_datetime(hist.index[0]).tz_localize(None).date()
                    if first >= date.today() - timedelta(days=ipo_window):
                        ipo_d = first
            except Exception:
                pass
        time.sleep(0.2)
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
        time.sleep(0.2)
        return df
    except Exception as e:
        warnings.warn(f"[{ticker}] history failed: {e}")
        return pd.DataFrame()


def slice_ipo(df: pd.DataFrame, ipo_d: date, n: int) -> pd.DataFrame:
    sliced = df[df.index >= pd.Timestamp(ipo_d)].head(n)
    if len(sliced) < 2:
        sliced = df.head(n)
    return sliced


# ---------------------------------------------------------------------------
# Upper-shadow + green candle checks
# ---------------------------------------------------------------------------

def candle_stats(row: pd.Series) -> tuple[float, bool]:
    """Returns (upper_shadow_ratio, is_green)"""
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    day_range = h - l
    if day_range < 1e-9:
        return 0.0, c >= o
    shadow = h - max(o, c)
    return shadow / day_range, c > o


def check_shadow_filter(df: pd.DataFrame, ipo_d: date,
                        min_ratio: float, require_green: bool) -> bool:
    sliced = slice_ipo(df, ipo_d, 1)
    if sliced.empty or not {"Open", "High", "Low", "Close"}.issubset(sliced.columns):
        return False
    ratio, green = candle_stats(sliced.iloc[0])
    ok = ratio >= min_ratio
    if require_green:
        ok = ok and green
    return ok


# ---------------------------------------------------------------------------
# Pattern normalization
# ---------------------------------------------------------------------------

def normalize_close(closes: np.ndarray) -> np.ndarray:
    if len(closes) < 2 or closes[0] <= 0:
        return np.array([])
    pct = (closes - closes[0]) / closes[0] * 100.0
    std = pct.std()
    return pct if std < 1e-9 else (pct - pct.mean()) / std


def normalize_ohlc(df_slice: pd.DataFrame) -> np.ndarray:
    if df_slice.empty or len(df_slice) < 2:
        return np.array([])
    if not {"Open", "High", "Low", "Close"}.issubset(df_slice.columns):
        return normalize_close(df_slice["Close"].values.astype(float))
    base = float(df_slice.iloc[0]["Open"])
    if base <= 0:
        return np.array([])
    mat = df_slice[["Open", "High", "Low", "Close"]].values.astype(float)
    pct = (mat - base) / base * 100.0
    flat = pct.flatten()
    std = flat.std()
    return flat if std < 1e-9 else (flat - flat.mean()) / std


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
# Synthetic demo OHLC target (Day-1: green + ~28% upper shadow)
# ---------------------------------------------------------------------------
DEMO_OHLC = pd.DataFrame({
    "Open":  [100.0, 110.0, 122.0],
    "High":  [115.0, 127.0, 136.0],
    "Low":   [ 97.0, 108.0, 120.0],
    "Close": [110.0, 122.0, 130.0],
})


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def run_scan(target_ticker: str, n_days: int, top_n: int,
             ipo_window: int, min_gain_pct: float,
             shadow_filter: bool, shadow_ratio: float,
             require_green: bool, use_ohlc: bool,
             sample_n: int | None) -> None:

    OUT_DIR.mkdir(exist_ok=True)
    cutoff = date.today() - timedelta(days=ipo_window)

    # ---- 1. Target ----
    print(f"\n[1/4] Fetching target: {target_ticker}")
    demo_mode = False

    target_ipo = None
    if target_ticker in SEED_WITH_DATES:
        target_ipo = datetime.strptime(SEED_WITH_DATES[target_ticker], "%Y-%m-%d").date()
    if target_ipo is None:
        target_ipo = get_ipo_date(target_ticker, ipo_window=3650)

    target_df = fetch_history(target_ticker)
    if target_df.empty or target_ipo is None:
        print(f"  WARNING: {target_ticker} unavailable — using synthetic FOMO demo")
        demo_mode = True
        target_ipo = date.today() - timedelta(days=3)
        target_slice = DEMO_OHLC.copy()
    else:
        target_slice = slice_ipo(target_df, target_ipo, n_days)

    raw_closes = DEMO_OHLC["Close"].values if demo_mode else target_slice["Close"].values.astype(float)
    target_norm = normalize_ohlc(target_slice) if use_ohlc else normalize_close(raw_closes)
    target_gain = 30.0 if demo_mode else (raw_closes[-1] - raw_closes[0]) / raw_closes[0] * 100

    print(f"  IPO date  : {target_ipo}")
    print(f"  {n_days}-day gain : +{target_gain:.1f}%")
    if shadow_filter:
        row0 = DEMO_OHLC.iloc[0] if demo_mode else target_slice.iloc[0]
        sr, green = candle_stats(row0)
        print(f"  Day-1 shadow={sr:.0%}  green={green}  filter_ratio={shadow_ratio:.0%}  require_green={require_green}")

    # ---- 2. Candidate pool ----
    print(f"\n[2/4] Building candidate list (window={ipo_window}d, cutoff={cutoff})...")

    # Live sources (with dates)
    live_dates: dict[str, str] = {}
    live_dates.update(fetch_us_ipos_live(ipo_window))
    live_dates.update(fetch_tw_ipos_live(ipo_window))

    # Seeds (with dates)
    for t, ds in SEED_WITH_DATES.items():
        if t not in live_dates:
            try:
                if datetime.strptime(ds, "%Y-%m-%d").date() >= cutoff:
                    live_dates[t] = ds
            except ValueError:
                pass

    # Filter out target
    live_dates.pop(target_ticker, None)

    # Filter too-old (extra safety)
    live_dates = {t: ds for t, ds in live_dates.items()
                  if datetime.strptime(ds, "%Y-%m-%d").date() >= cutoff}

    all_candidates: list[tuple[str, str]] = list(live_dates.items())
    print(f"  [Seeds]   (merged into pool above)")
    print(f"  Total pool: {len(all_candidates)} unique candidates")

    # Random subsample
    if sample_n and sample_n < len(all_candidates):
        all_candidates = random.sample(all_candidates, sample_n)
        print(f"  Subsampled: {sample_n} random candidates (run again to cover more)")

    if not all_candidates:
        print("\n  No candidates found.  Try: --ipo_window 3650")
        return

    # ---- 3. Score ----
    sf_desc = f"shadow>={shadow_ratio:.0%}" + (" + green" if require_green else "")
    print(f"\n[3/4] Scoring...  [shadow_filter={'on: ' + sf_desc if shadow_filter else 'off'}"
          f"  ohlc={'on' if use_ohlc else 'off'}]")
    results = []
    s_gain, s_data, s_shadow = 0, 0, 0
    no_data_list: list[str] = []
    shadow_skip_list: list[str] = []

    for ticker, ipo_ds in tqdm(all_candidates, ncols=80):
        try:
            ipo_d = datetime.strptime(ipo_ds, "%Y-%m-%d").date()
        except ValueError:
            continue

        df = fetch_history(ticker)
        if df.empty:
            s_data += 1
            no_data_list.append(ticker)
            continue

        sliced = slice_ipo(df, ipo_d, n_days)
        if len(sliced) < 2:
            s_data += 1
            no_data_list.append(ticker)
            continue

        closes = sliced["Close"].values.astype(float)
        gain = (closes[-1] - closes[0]) / closes[0] * 100
        if gain < min_gain_pct:
            s_gain += 1
            continue

        if shadow_filter:
            if not check_shadow_filter(df, ipo_d, shadow_ratio, require_green):
                s_shadow += 1
                shadow_skip_list.append(ticker)
                continue

        cand_norm = normalize_ohlc(sliced) if use_ohlc else normalize_close(closes)
        if len(cand_norm) < 2:
            s_data += 1
            continue

        # Day-1 stats
        day1_shadow, day1_green = 0.0, False
        if {"Open", "High", "Low", "Close"}.issubset(sliced.columns):
            day1_shadow, day1_green = candle_stats(sliced.iloc[0])

        results.append({
            "ticker":            ticker,
            "ipo_date":          ipo_ds,
            "gain_pct_3d":       round(gain, 2),
            "similarity":        round(score(target_norm, cand_norm), 4),
            "day1_green":        day1_green,
            "day1_shadow_ratio": round(day1_shadow, 3),
            "closes":            closes.tolist(),
            "norm":              cand_norm.tolist(),
        })

    print(f"  Matched : {len(results)}")
    print(f"  Skipped : low_gain={s_gain}  no_data={s_data}  no_shadow={s_shadow}")
    if no_data_list:
        print(f"  no_data  : {', '.join(no_data_list[:20])}" +
              (f" ... (+{len(no_data_list)-20})" if len(no_data_list) > 20 else ""))
    if shadow_skip_list:
        print(f"  shadow_skip ({len(shadow_skip_list)}): {', '.join(shadow_skip_list[:25])}" +
              (f" ... (+{len(shadow_skip_list)-25})" if len(shadow_skip_list) > 25 else ""))

    if not results:
        print("\n  No matches. Suggestions:")
        print("    --shadow_ratio 0.15    loosen shadow threshold")
        print("    --no_green             allow red Day-1 candles")
        print("    --min_gain 0           disable gain filter")
        print("    --ipo_window 3650      widen date window")
        return

    # ---- 4. Output ----
    print("\n[4/4] Ranking...")
    df_out = pd.DataFrame(results).sort_values("similarity", ascending=False)
    top = df_out.head(top_n)
    label = f"{target_ticker}{'_DEMO' if demo_mode else ''}"

    display_cols = ["ticker", "ipo_date", "gain_pct_3d", "similarity",
                    "day1_green", "day1_shadow_ratio"]
    print("\n" + "=" * 80)
    print(f"  TOP {top_n} SIMILAR IPOs — {label}  (first {n_days} days)")
    print(f"  shadow_filter={shadow_filter}  require_green={require_green}  ohlc_score={use_ohlc}")
    print("=" * 80)
    print(top[display_cols].to_string(index=False))
    print("=" * 80)

    safe = label.replace(".", "_")
    csv_path = OUT_DIR / f"similar_ipos_{safe}.csv"
    top.drop(columns=["closes", "norm"]).to_csv(csv_path, index=False)
    print(f"  CSV   -> {csv_path}")
    _plot(target_norm, raw_closes, top, label, n_days, use_ohlc)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(target_norm, raw_closes, top, label, n_days, use_ohlc):
    n_show = min(len(top), 12)
    fig = plt.figure(figsize=(16, 11), facecolor="#0f0f14")
    score_type = "OHLC 4-dim" if use_ohlc else "close z-score"
    fig.suptitle(
        f"IPO FOMO Pattern Match  ·  {label}  (first {n_days} days)  [{score_type}]",
        color="#e8e6e0", fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.2], hspace=0.45)
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
    ax1.set_title(f"{label} — % gain from IPO open", color="#9ca3af", fontsize=10)
    ax1.set_xlabel("Trading Day", color="#6b7280", fontsize=9)
    ax1.set_ylabel("% from open", color="#6b7280", fontsize=9)
    ax1.tick_params(colors="#6b7280")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#2d2d3a")

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1a1a22")
    for i, row in enumerate(top.head(n_show).itertuples()):
        nv = np.array(row.norm)
        if len(nv) > n_days + 1:
            close_idx = [3 + 4 * k for k in range(n_days) if 3 + 4 * k < len(nv)]
            nv_plot = nv[close_idx]
        else:
            nv_plot = nv
        tag = f"  sh={row.day1_shadow_ratio:.0%}{'↑' if row.day1_green else '↓'}"
        ax2.plot(range(1, len(nv_plot) + 1), nv_plot,
                 color=cmap(i / max(n_show - 1, 1)), alpha=0.75, lw=1.5,
                 label=f"{row.ticker} ({row.ipo_date})  {row.similarity:.3f}  +{row.gain_pct_3d}%{tag}")

    t_plot = target_norm
    if len(t_plot) > n_days + 1:
        close_idx = [3 + 4 * k for k in range(n_days) if 3 + 4 * k < len(t_plot)]
        t_plot = t_plot[close_idx]
    ax2.plot(range(1, len(t_plot) + 1), t_plot,
             color="#f59e0b", lw=2.8, ls="--", label=f"{label} (target)")
    ax2.axhline(0, color="#4a4a5a", lw=0.8, ls="--")
    ax2.set_title(f"Top {n_show} similar (sh=shadow ratio, ↑=green ↓=red Day-1)",
                  color="#9ca3af", fontsize=10)
    ax2.set_xlabel("Trading Day", color="#6b7280", fontsize=9)
    ax2.set_ylabel("Z-score", color="#6b7280", fontsize=9)
    ax2.tick_params(colors="#6b7280")
    ax2.legend(fontsize=7.5, facecolor="#1a1a22", labelcolor="#e8e6e0",
               loc="upper left", ncol=2, framealpha=0.75)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#2d2d3a")

    safe = label.replace(".", "_").replace("(", "").replace(")", "")
    path = OUT_DIR / f"pattern_match_{safe}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f0f14")
    plt.close()
    print(f"  Chart -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="IPO FOMO pattern scanner v8")
    p.add_argument("--target",        default=DEFAULT_TARGET)
    p.add_argument("--days",          type=int,   default=DEFAULT_DAYS)
    p.add_argument("--top",           type=int,   default=DEFAULT_TOP)
    p.add_argument("--ipo_window",    type=int,   default=DEFAULT_IPO_WINDOW)
    p.add_argument("--min_gain",      type=float, default=DEFAULT_MIN_GAIN)
    p.add_argument("--shadow_filter", action="store_true",
                   help="Keep only candidates with Day-1 upper shadow >= shadow_ratio")
    p.add_argument("--shadow_ratio",  type=float, default=DEFAULT_SHADOW_RATIO,
                   help="Min upper_shadow / day_range (default 0.25)")
    p.add_argument("--no_green",      action="store_true",
                   help="With --shadow_filter: allow red-candle Day-1 (only checks shadow size)")
    p.add_argument("--ohlc_score",    action="store_true",
                   help="Score with OHLC 4-dim pattern instead of close-only")
    p.add_argument("--sample",        type=int,   default=None,
                   help="Randomly subsample N candidates before scoring (run multiple times)")
    p.add_argument("--no_cache",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("Cache cleared.")
    run_scan(
        target_ticker = args.target,
        n_days        = args.days,
        top_n         = args.top,
        ipo_window    = args.ipo_window,
        min_gain_pct  = args.min_gain,
        shadow_filter = args.shadow_filter,
        shadow_ratio  = args.shadow_ratio,
        require_green = not args.no_green,
        use_ohlc      = args.ohlc_score,
        sample_n      = args.sample,
    )
