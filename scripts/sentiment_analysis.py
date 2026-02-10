"""
Sentiment analysis for stock news headlines using FinBERT.

Prerequisites:
1. Download the Kaggle dataset from:
   https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
2. Extract to data/kaggle_news/ (or update INPUT_PATH below)
3. Install dependencies: pip install transformers torch pandas tqdm

Usage:
    python scripts/sentiment_analysis.py
"""

import sys
from pathlib import Path

# -----------------------------
# Fail-fast checks (run these BEFORE slow imports)
# -----------------------------
def preflight_checks():
    """Run all checks upfront so we fail fast, not after 30 min."""
    errors = []

    # 1. Check Python version
    if sys.version_info < (3, 9):
        errors.append(f"Python 3.9+ required, got {sys.version_info.major}.{sys.version_info.minor}")

    # 2. Check required packages
    missing_packages = []
    for pkg in ["torch", "transformers", "pandas", "tqdm"]:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)
    if missing_packages:
        errors.append(f"Missing packages: {', '.join(missing_packages)}\n   Run: pip install {' '.join(missing_packages)}")

    # 3. Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        elif torch.backends.mps.is_available():
            print("✓ GPU: Apple Silicon (MPS)")
        else:
            print("⚠ Warning: No GPU detected, using CPU (will be slow)")
    except Exception as e:
        errors.append(f"GPU check failed: {e}")

    # 4. Check input file exists
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "kaggle_news" / "analyst_ratings_processed.csv"
    if not input_path.exists():
        errors.append(
            f"Input file not found: {input_path}\n"
            f"   Download from: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests\n"
            f"   Extract 'analyst_ratings_processed.csv' to: {input_path.parent}/"
        )
    else:
        # Check file is readable and has expected columns
        try:
            import pandas as pd
            df_sample = pd.read_csv(input_path, nrows=5)
            expected_cols = {"title", "date", "stock"}
            actual_cols = set(df_sample.columns)
            if not expected_cols.issubset(actual_cols):
                errors.append(f"Input file missing columns. Expected: {expected_cols}, Got: {actual_cols}")
            else:
                # Count total rows (fast estimate)
                with open(input_path, 'r') as f:
                    row_count = sum(1 for _ in f) - 1
                print(f"✓ Input file: {input_path.name} ({row_count:,} rows)")
        except Exception as e:
            errors.append(f"Cannot read input file: {e}")

    # 5. Check output directory is writable
    output_dir = project_root / "data"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        print(f"✓ Output directory writable: {output_dir}")
    except Exception as e:
        errors.append(f"Cannot write to output directory {output_dir}: {e}")

    # 6. Test FinBERT model loads
    try:
        from transformers import pipeline
        print("✓ Transformers loaded, testing FinBERT...")
        test_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU for quick test
        )
        result = test_pipeline("Stock prices rose today")
        print(f"✓ FinBERT working: test result = {result[0]['label']}")
        del test_pipeline
    except Exception as e:
        errors.append(f"FinBERT model failed to load: {e}")

    # Report results
    if errors:
        print("\n" + "="*50)
        print("PREFLIGHT CHECKS FAILED:")
        print("="*50)
        for i, err in enumerate(errors, 1):
            print(f"\n{i}. {err}")
        print("\n" + "="*50)
        sys.exit(1)
    else:
        print("\n✓ All preflight checks passed!\n")


# Run checks immediately
preflight_checks()

# Now do slow imports
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input: Kaggle dataset (download manually)
INPUT_PATH = PROJECT_ROOT / "data" / "kaggle_news" / "analyst_ratings_processed.csv"

# Output: Sentiment scores aggregated by (date, ticker)
OUTPUT_PATH = PROJECT_ROOT / "data" / "news_sentiment.csv"

# Checkpoint file (saves progress, can resume if interrupted)
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "news_sentiment_checkpoint.csv"

# Save checkpoint every N batches
CHECKPOINT_EVERY = 50  # ~12,800 headlines with BATCH_SIZE=256

# Date range to filter (to match seb_output.csv)
START_DATE = "2016-01-01"
END_DATE = "2020-12-31"

# Batch size for FinBERT (adjust based on your GPU/RAM)
# RTX 6000 (24GB) can handle 256-512, smaller GPUs use 64
BATCH_SIZE = 256

# Set to True to do a quick test run on 1000 rows
TEST_MODE = False


def load_and_filter_news(path: Path, start: str, end: str) -> pd.DataFrame:
    """Load news data and filter to date range."""
    print(f"Loading news from {path}...")
    df = pd.read_csv(path)

    # The Kaggle dataset has columns: title, date, stock
    # Rename for consistency
    df = df.rename(columns={"title": "headline", "stock": "ticker"})

    # Parse date (format: "2020-01-15 09:30:00-04:00" or similar)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.date

    # Filter date range
    df = df[(df["date"] >= pd.to_datetime(start).date()) &
            (df["date"] <= pd.to_datetime(end).date())]

    # Drop rows with missing headlines
    df = df.dropna(subset=["headline"])
    df = df[df["headline"].str.strip() != ""]

    print(f"Loaded {len(df):,} headlines from {start} to {end}")
    return df


def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA for NVIDIA, else CPU)."""
    import torch

    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return "mps"
    elif torch.cuda.is_available():
        print("Using NVIDIA GPU (CUDA)")
        return 0
    else:
        print("Using CPU")
        return -1


def run_sentiment_analysis(
    df: pd.DataFrame,
    batch_size: int = 64,
    checkpoint_every: int = 50,
    checkpoint_path: Path = None,
) -> pd.DataFrame:
    """Run FinBERT sentiment analysis with checkpointing and validation."""
    from transformers import pipeline

    device = get_device()
    headlines = df["headline"].tolist()

    # Check for existing checkpoint
    start_idx = 0
    results = []
    if checkpoint_path and checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}")
        checkpoint_df = pd.read_csv(checkpoint_path)
        start_idx = len(checkpoint_df)
        results = checkpoint_df[["label", "confidence"]].to_dict("records")
        results = [{"label": r["label"], "score": r["confidence"]} for r in results]
        print(f"Resuming from row {start_idx:,} / {len(headlines):,}")

    if start_idx >= len(headlines):
        print("Already completed!")
        df["label"] = [r["label"] for r in results]
        df["confidence"] = [r["score"] for r in results]
        return df

    print("Loading FinBERT model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=device,
    )

    # Quick sanity test
    test_result = sentiment_pipeline("Stock prices increased significantly")
    print(f"Sanity test: 'Stock prices increased significantly' -> {test_result[0]['label']}")
    if test_result[0]["label"] != "positive":
        print("⚠ Warning: Model may not be working correctly")

    print(f"Analyzing {len(headlines) - start_idx:,} remaining headlines...")
    batches_processed = 0
    failed_count = 0

    pbar = tqdm(range(start_idx, len(headlines), batch_size), desc="Processing")
    for i in pbar:
        batch = headlines[i:i + batch_size]
        batch = [str(h)[:512] if h and len(str(h)) > 512 else str(h) if h else "" for h in batch]

        try:
            batch_results = sentiment_pipeline(batch)
            results.extend(batch_results)
        except Exception as e:
            print(f"\n⚠ Batch {i} failed: {e}")
            # Process one by one
            for h in batch:
                try:
                    results.append(sentiment_pipeline(h[:512] if h else "")[0])
                except:
                    results.append({"label": "neutral", "score": 0.5})
                    failed_count += 1

        batches_processed += 1

        # Save checkpoint periodically
        if checkpoint_path and batches_processed % checkpoint_every == 0:
            _save_checkpoint(df, results, checkpoint_path)
            pbar.set_postfix({"saved": len(results), "failed": failed_count})

    # Final checkpoint
    if checkpoint_path:
        _save_checkpoint(df, results, checkpoint_path)

    if failed_count > 0:
        print(f"⚠ {failed_count:,} headlines failed, defaulted to neutral")

    # Validate results
    _validate_results(results)

    df["label"] = [r["label"] for r in results]
    df["confidence"] = [r["score"] for r in results]
    return df


def _save_checkpoint(df: pd.DataFrame, results: list[dict], checkpoint_path: Path):
    """Save current progress to checkpoint file."""
    temp_df = df.iloc[:len(results)].copy()
    temp_df["label"] = [r["label"] for r in results]
    temp_df["confidence"] = [r["score"] for r in results]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_df.to_csv(checkpoint_path, index=False)


def _validate_results(results: list[dict]):
    """Check that results look reasonable."""
    labels = [r["label"] for r in results]
    total = len(labels)

    pos = labels.count("positive") / total * 100
    neg = labels.count("negative") / total * 100
    neu = labels.count("neutral") / total * 100

    print(f"\n--- Results Validation ---")
    print(f"Total headlines: {total:,}")
    print(f"Positive: {pos:.1f}%")
    print(f"Negative: {neg:.1f}%")
    print(f"Neutral:  {neu:.1f}%")

    # Sanity checks
    if pos > 90 or neg > 90 or neu > 90:
        print("⚠ WARNING: Results are heavily skewed - something may be wrong!")
    elif pos < 1 or neg < 1:
        print("⚠ WARNING: Almost no positive/negative - model may not be working!")
    else:
        print("✓ Distribution looks reasonable")


def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment scores by (date, ticker)."""
    # Convert FinBERT labels to numeric scores
    # FinBERT outputs: positive, negative, neutral
    label_to_score = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_score"] = df["label"].map(label_to_score) * df["confidence"]

    # Aggregate by date and ticker
    agg = df.groupby(["date", "ticker"]).agg(
        news_count=("headline", "count"),
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_sum=("sentiment_score", "sum"),
        positive_count=("label", lambda x: (x == "positive").sum()),
        negative_count=("label", lambda x: (x == "negative").sum()),
        neutral_count=("label", lambda x: (x == "neutral").sum()),
    ).reset_index()

    # Add sentiment ratio (positive - negative) / total
    agg["sentiment_ratio"] = (agg["positive_count"] - agg["negative_count"]) / agg["news_count"]

    return agg


def main():
    # Check input file exists
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")
        print(f"\nThen extract 'analyst_ratings_processed.csv' to: {INPUT_PATH.parent}/")
        return

    # Load and filter news
    df = load_and_filter_news(INPUT_PATH, START_DATE, END_DATE)

    if TEST_MODE:
        print("TEST MODE: Using only 1000 rows")
        df = df.head(1000)

    if len(df) == 0:
        print("No headlines found in date range!")
        return

    # Run sentiment analysis with checkpointing
    df = run_sentiment_analysis(
        df,
        batch_size=BATCH_SIZE,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # Aggregate by (date, ticker)
    print("Aggregating sentiment by (date, ticker)...")
    agg = aggregate_sentiment(df)

    # Validate aggregated output
    print("\n--- Aggregated Output Validation ---")
    if agg["sentiment_mean"].isna().all():
        print("ERROR: All sentiment values are NaN!")
        return
    if len(agg) < 100:
        print(f"⚠ WARNING: Only {len(agg)} rows - expected more")
    if agg["ticker"].nunique() < 10:
        print(f"⚠ WARNING: Only {agg['ticker'].nunique()} unique tickers")
    else:
        print(f"✓ {len(agg):,} rows, {agg['ticker'].nunique()} tickers")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved aggregated sentiment to: {OUTPUT_PATH}")
    print(f"  Shape: {agg.shape}")
    print(f"  Date range: {agg['date'].min()} to {agg['date'].max()}")
    print(f"  Unique tickers: {agg['ticker'].nunique()}")

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"✓ Cleaned up checkpoint file")

    print(f"\nSample output:")
    print(agg.head(10))


if __name__ == "__main__":
    main()
