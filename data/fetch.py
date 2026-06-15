"""
fetch.py
========
下載並快取 OHLCV 資料（yfinance）。
快取存在 data/cache/{SYMBOL}_{start}_{end}.parquet。
"""

from __future__ import annotations

import datetime
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
    pd.DataFrame, index=DatetimeIndex, columns: Open High Low Close Volume
    """
    cache_path = CACHE_DIR / f"{symbol}_{start}_{end}.parquet"

    if cache_path.exists() and not force_download:
        df = pd.read_parquet(cache_path)
        print(f"[fetch] cache hit: {cache_path.name}")
        # 確保 index 是 DatetimeIndex
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
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

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # 存 parquet（把 index 存進去）
    df.to_parquet(cache_path, index=True)
    print(f"[fetch] saved to {cache_path.name}  ({len(df)} rows)")
    return df


def get_ohlcv(
    symbol: str,
    start:  str | None = None,
    end:    str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    run_sim.py 使用的入口。start/end 可省略，預設抓近兩年。

    Returns
    -------
    pd.DataFrame, index=DatetimeIndex
    """
    if end is None:
        end = datetime.date.today().isoformat()
    if start is None:
        start = (datetime.date.today() - datetime.timedelta(days=730)).isoformat()
    print(f"[fetch] range: {start} ~ {end}")
    return fetch_ohlcv(symbol, start=start, end=end, force_download=force_download)
