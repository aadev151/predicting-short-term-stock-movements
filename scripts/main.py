import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from pathlib import Path


def get_sp500_tickers() -> tuple[list[str], list[str], dict[str, str]]:
    """Return (yf_tickers, original_tickers, yf_to_original) for current S&P 500.

    Wikipedia uses tickers like BRK.B, BF.B; yfinance expects BRK-B, BF-B.
    We fetch the page with a browser-like User-Agent to avoid 403 blocks.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    table = tables[0]

    original = table["Symbol"].astype(str).tolist()
    yf_tickers = [t.replace(".", "-") for t in original]
    yf_to_original = {t.replace(".", "-"): t for t in original}
    return yf_tickers, original, yf_to_original


# -----------------------------
# Config
# -----------------------------
# Pull ~10 years of history from today
START_DATE = (pd.Timestamp.today().normalize() - pd.DateOffset(years=10)).date().isoformat()
END_DATE = None  # None = today

# Save relative to this script: scripts/ -> project root -> data/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "data" / "sp500_simple.csv"

# Use an index ticker to define the trading-day calendar
CAL_TICKER = "^GSPC"


# -----------------------------
# Download data (current S&P 500 constituents)
# -----------------------------
# NOTE: This intentionally uses the CURRENT S&P 500 list even for older dates.
# That introduces survivorship bias (as you requested).

yf_tickers, original_tickers, yf_to_original = get_sp500_tickers()

raw = yf.download(
    tickers=yf_tickers,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    group_by="ticker",
    threads=True,
)

# Ensure columns are (ticker, field)
if not isinstance(raw.columns, pd.MultiIndex):
    raise ValueError("Expected MultiIndex columns from yfinance when downloading multiple tickers.")

lvl0 = set(raw.columns.get_level_values(0))
lvl1 = set(raw.columns.get_level_values(1))
tset = set(yf_tickers)
# If tickers live in level 1, swap
if len(lvl1 & tset) > len(lvl0 & tset):
    raw = raw.swaplevel(0, 1, axis=1)

raw = raw.sort_index(axis=1)


# -----------------------------
# Reshape to tidy (date, ticker, fields)
# -----------------------------
df = (
    raw.stack(level=0)
       .rename_axis(index=["date", "ticker"])
       .reset_index()
)

# Rename fields
# (yfinance uses 'Adj Close' with a space)
df.rename(
    columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "AdjClose": "adj_close",
        "Volume": "volume",
    },
    inplace=True,
)

# Convert yfinance ticker back to the Wikipedia/original ticker format (BRK-B -> BRK.B)
df["ticker"] = df["ticker"].map(lambda t: yf_to_original.get(t, t))


# -----------------------------
# Force 500 rows per trading date (full date x ticker grid)
# -----------------------------
cal = yf.download(CAL_TICKER, start=START_DATE, end=END_DATE, auto_adjust=False)
trading_dates = cal.index

full_index = pd.MultiIndex.from_product(
    [trading_dates, original_tickers],
    names=["date", "ticker"],
)

# Reindex (adds missing rows as NaNs for tickers that didn't trade / didn't exist yet)
df = (
    df.set_index(["date", "ticker"]).reindex(full_index).reset_index()
)


# -----------------------------
# Features (per ticker)
# -----------------------------
df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")

df = df.sort_values(["ticker", "date"])  # important for rolling/pct_change

# Simple returns
df["daily_return"] = df.groupby("ticker")["adj_close"].pct_change()

# Log returns (use transform to keep index aligned)
ratio = df.groupby("ticker")["adj_close"].transform(lambda s: s / s.shift(1))
df["log_return"] = np.log(ratio.where(ratio > 0))

# Rolling stats
df["rolling_mean_20"] = df.groupby("ticker")["adj_close"].transform(lambda s: s.rolling(20).mean())
df["rolling_std_20"] = df.groupby("ticker")["adj_close"].transform(lambda s: s.rolling(20).std())


# -----------------------------
# Save
# -----------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(f"Saved dataset to {OUT_PATH.resolve()}")