"""
fetch.py
========
下載並快取 OHLCV 資料（yfinance）。
快取存在 data/cache/{SYMBOL}_{start}_{end}.parquet。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def fetch_ohlcv(
    symbol: str,
    start:  str,
    end:    str,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    下載並快取 OHLCV。

    Parameters
    ----------
    symbol  : 股票代碼，例如 'AAPL'
    start   : 開始日期 'YYYY-MM-DD'
    end     : 結束日期 'YYYY-MM-DD'
    force_download : True 強制重新下載

    Returns
    -------
    pd.DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    cache_path = CACHE_DIR / f"{symbol}_{start}_{end}.parquet"

    if cache_path.exists() and not force_download:
        df = pd.read_parquet(cache_path)
        print(f"[fetch] cache hit: {cache_path.name}")
        return df

    print(f"[fetch] downloading {symbol} {start} ~ {end} ...")
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"index": "Date"}) if "index" in df.columns else df

    df.to_parquet(cache_path, index=False)
    print(f"[fetch] saved to {cache_path.name}  ({len(df)} rows)")
    return df
