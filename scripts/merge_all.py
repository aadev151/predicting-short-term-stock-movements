"""
Merge all preliminary datasets into one unified DataFrame.

Each dataset in data/preliminary_merge_results/ was originally merged with
data/sp500_simple.csv. This script starts from sp500_simple as the base and
merges only the NEW columns from each preliminary result.

Datasets & their unique columns:
  - sp500_combined.csv (Umesh):  company metadata + S&P500 index
  - nr_combined_stock_data.csv (Naziia): NYSE open/high/low/close
  - market_regimes_merged.csv (Seb): VIX, Yield_Spread, Regime_GMM, Regime_label
  - news_sentiment.csv (Alex): news counts & sentiment scores
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRELIM_DIR = DATA_DIR / "preliminary_merge_results"
OUTPUT_PATH = DATA_DIR / "merged_dataset.csv"

BASE_COLS = [
    "date", "ticker", "adj_close", "close", "high", "low", "open",
    "volume", "daily_return", "log_return", "rolling_mean_20", "rolling_std_20",
]


def load_base():
    """Load sp500_simple.csv as the base DataFrame."""
    df = pd.read_csv(DATA_DIR / "sp500_simple.csv", parse_dates=["date"])
    print(f"Base (sp500_simple): {df.shape}")
    return df


def merge_sp500_combined(base):
    """
    Merge company metadata (ticker-level) and S&P 500 index price (date-level)
    from Umesh's sp500_combined.csv.
    """
    df = pd.read_csv(PRELIM_DIR / "sp500_combined.csv", parse_dates=["date"])

    # --- Company metadata: one row per ticker ---
    company_cols = [
        "Exchange", "Shortname", "Longname", "Sector", "Industry",
        "Currentprice", "Marketcap", "Ebitda", "Revenuegrowth",
        "City", "State", "Country", "Fulltimeemployees",
        "Longbusinesssummary", "Weight",
    ]
    companies = (
        df[["ticker"] + company_cols]
        .drop_duplicates(subset=["ticker"])
    )

    # --- S&P 500 index price: one row per date ---
    index_cols = ["S&P500"]
    index_data = (
        df[["date"] + index_cols]
        .drop_duplicates(subset=["date"])
    )

    merged = base.merge(companies, on="ticker", how="left")
    merged = merged.merge(index_data, on="date", how="left")
    print(f"After sp500_combined merge: {merged.shape}")
    return merged


def merge_nyse_data(base):
    """
    Merge NYSE open/high/low/close from Naziia's nr_combined_stock_data.csv.
    """
    df = pd.read_csv(PRELIM_DIR / "nr_combined_stock_data.csv", parse_dates=["date"])

    nyse_cols = ["open_nr", "high_nr", "low_nr", "close_nr"]
    nyse = df[["date", "ticker"] + nyse_cols].copy()

    merged = base.merge(nyse, on=["date", "ticker"], how="left")
    print(f"After NYSE merge: {merged.shape}")
    return merged


def merge_market_regimes(base):
    """
    Merge VIX, yield spread, and GMM regime labels from Seb's
    market_regimes_merged.csv. These are date-level (same for all tickers).
    """
    df = pd.read_csv(PRELIM_DIR / "market_regimes_merged.csv", parse_dates=["date"])

    regime_cols = ["VIX", "Yield_Spread", "Regime_GMM", "Regime_label"]
    regimes = (
        df[["date"] + regime_cols]
        .drop_duplicates(subset=["date"])
    )

    merged = base.merge(regimes, on="date", how="left")
    print(f"After market regimes merge: {merged.shape}")
    return merged


def merge_news_sentiment(base):
    """
    Merge news sentiment features from Alex's news_sentiment.csv.
    """
    df = pd.read_csv(PRELIM_DIR / "news_sentiment.csv", parse_dates=["date"])

    sentiment_cols = [
        "news_count", "sentiment_mean", "sentiment_sum",
        "positive_count", "negative_count", "neutral_count", "sentiment_ratio",
    ]
    sentiment = df[["date", "ticker"] + sentiment_cols].copy()

    merged = base.merge(sentiment, on=["date", "ticker"], how="left")
    print(f"After news sentiment merge: {merged.shape}")
    return merged


def main():
    df = load_base()
    df = merge_sp500_combined(df)
    df = merge_nyse_data(df)
    df = merge_market_regimes(df)
    df = merge_news_sentiment(df)

    print(f"\nFinal dataset: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
