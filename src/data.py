"""Data acquisition for the conditional weekend-effect study.

Pulls daily prices for SPY, the nine SPDR sector ETFs, and the CBOE VIX
index from Yahoo Finance (no API key required), then builds a tidy panel of
daily log returns aligned with the contemporaneous closing VIX level.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yfinance as yf

# SPDR sector ETFs. XLC (Communication Services) launched mid-2018, so it is
# included but will simply have a shorter usable sample.
SECTOR_ETFS: dict[str, str] = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

MARKET = "SPY"
VIX = "^VIX"

DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-01-01"

_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "panel.csv")


def _download_closes(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closes for a list of tickers as a wide frame."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    # With multiple tickers yfinance returns a column MultiIndex (field, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:  # single ticker
        closes = raw[["Close"]].copy()
        closes.columns = tickers
    return closes


def _download_ohlc(
    tickers: list[str], start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted Open and Close for a list of tickers as wide frames."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Open"].copy(), raw["Close"].copy()
    opens = raw[["Open"]].copy()
    closes = raw[["Close"]].copy()
    opens.columns = tickers
    closes.columns = tickers
    return opens, closes


_GAP_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "panel_gap.csv")


def build_gap_panel(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    use_cache: bool = True,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Long panel decomposing each day's return into overnight + intraday parts.

    For trading day t:
        overnight_t = log(Open_t  / Close_{t-1}) * 100   (incl. the weekend gap on Mondays)
        intraday_t  = log(Close_t / Open_t)      * 100
        total_t     = overnight_t + intraday_t = log(Close_t / Close_{t-1}) * 100

    Conditioning VIX is the prior-day close (no look-ahead), as in build_panel.
    """
    cache = cache_path or _GAP_CACHE
    if use_cache and os.path.exists(cache):
        cached = pd.read_csv(cache, parse_dates=["date"])
        cached["weekday"] = pd.Categorical(
            cached["weekday"], categories=["Mon", "Tue", "Wed", "Thu", "Fri"],
            ordered=True,
        )
        return cached

    price_tickers = [MARKET, *SECTOR_ETFS.keys()]
    opens, closes = _download_ohlc(price_tickers, start, end)
    _, vix = _download_ohlc([VIX], start, end)
    vix.columns = ["vix"]

    overnight = np.log(opens / closes.shift(1)) * 100.0
    intraday = np.log(closes / opens) * 100.0

    labels = {MARKET: "Market", **SECTOR_ETFS}
    frames = []
    for ticker, label in labels.items():
        if ticker not in closes.columns:
            continue
        on = overnight[ticker]
        ind = intraday[ticker]
        df = pd.DataFrame(
            {
                "date": on.index,
                "asset": ticker,
                "sector": label,
                "overnight": on.values,
                "intraday": ind.values,
            }
        ).dropna(subset=["overnight", "intraday"])
        df["total"] = df["overnight"] + df["intraday"]
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["weekday"] = pd.Categorical(
        panel["date"].dt.day_name().str[:3],
        categories=["Mon", "Tue", "Wed", "Thu", "Fri"],
        ordered=True,
    )
    vix_lag = vix["vix"].shift(1).rename("vix")
    panel = panel.merge(vix_lag, left_on="date", right_index=True, how="left")
    panel = panel.dropna(subset=["weekday"])

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    panel.to_csv(cache, index=False)
    return panel


def build_panel(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    use_cache: bool = True,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Return a long panel: one row per (date, asset) with return, weekday, VIX.

    Columns
    -------
    date        : trading day (Timestamp)
    asset       : ticker (SPY or a sector ETF)
    sector      : human-readable label ("Market" for SPY)
    ret         : daily log return * 100 (i.e. in percent)
    weekday     : 'Mon'..'Fri'
    vix         : closing VIX level on that date
    """
    cache = cache_path or _CACHE
    if use_cache and os.path.exists(cache):
        cached = pd.read_csv(cache, parse_dates=["date"])
        cached["weekday"] = pd.Categorical(
            cached["weekday"], categories=["Mon", "Tue", "Wed", "Thu", "Fri"],
            ordered=True,
        )
        return cached

    price_tickers = [MARKET, *SECTOR_ETFS.keys()]
    prices = _download_closes(price_tickers, start, end)
    vix = _download_closes([VIX], start, end)
    vix.columns = ["vix"]

    # Daily log returns in percent.
    rets = np.log(prices / prices.shift(1)) * 100.0
    rets = rets.dropna(how="all")

    labels = {MARKET: "Market", **SECTOR_ETFS}

    frames = []
    for ticker, label in labels.items():
        if ticker not in rets.columns:
            continue
        s = rets[ticker].dropna()
        df = pd.DataFrame(
            {
                "date": s.index,
                "asset": ticker,
                "sector": label,
                "ret": s.values,
            }
        )
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["weekday"] = pd.Categorical(
        panel["date"].dt.day_name().str[:3],
        categories=["Mon", "Tue", "Wed", "Thu", "Fri"],
        ordered=True,
    )
    # Use the PRIOR day's VIX close as the conditioning variable: it is known
    # before the trading day begins, so the regime split is not look-ahead.
    vix_lag = vix["vix"].shift(1).rename("vix")
    panel = panel.merge(vix_lag, left_on="date", right_index=True, how="left")
    panel = panel.dropna(subset=["ret", "vix", "weekday"])

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    panel.to_csv(cache, index=False)
    return panel


if __name__ == "__main__":
    p = build_panel(use_cache=False)
    print(p.head())
    print(f"\nrows={len(p):,}  assets={p['asset'].nunique()}  "
          f"dates={p['date'].nunique():,}  "
          f"range={p['date'].min().date()}..{p['date'].max().date()}")
