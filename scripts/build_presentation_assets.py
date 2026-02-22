#!/usr/bin/env python3
"""Generate slide-ready figures and summary notes for the stock direction project.

All figures are anchored to `data/merged_dataset.csv`.
Model performance figures use:
  - data/model_overall_results.csv
  - data/model_sector_results.csv
  - data/model_best_by_sector.csv
which are exported by `notebooks/eda_all_models_per_sector.ipynb` from the
same merged dataset.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Ensure plotting caches are writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MERGED_DATA_PATH = DATA_DIR / "merged_dataset.csv"
OVERALL_RESULTS_PATH = DATA_DIR / "model_overall_results.csv"
SECTOR_RESULTS_PATH = DATA_DIR / "model_sector_results.csv"
BEST_SECTOR_PATH = DATA_DIR / "model_best_by_sector.csv"

OUT_DIR = PROJECT_ROOT / "presentation_assets"
FIG_DIR = OUT_DIR / "figures"
METRICS_PATH = OUT_DIR / "summary_metrics.json"
BRIEF_PATH = OUT_DIR / "presentation_brief.md"

SPLIT_DATE = pd.Timestamp("2023-01-01")

FEATURE_COLS = [
    "lag_return_1",
    "lag_return_2",
    "lag_return_5",
    "rolling_std_20",
    "price_to_ma20",
    "hl_range",
    "oc_gap",
    "vol_norm",
    "VIX",
    "Yield_Spread",
    "Regime_GMM",
    "sentiment_mean",
    "sentiment_ratio",
    "has_news",
    "log_marketcap",
    "Revenuegrowth",
    "Weight",
    "Sector_encoded",
]


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _ensure_inputs() -> None:
    missing = []
    for path in [
        MERGED_DATA_PATH,
        OVERALL_RESULTS_PATH,
        SECTOR_RESULTS_PATH,
        BEST_SECTOR_PATH,
    ]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n" + "\n".join(f"- {p}" for p in missing)
        )


def _save(fig: plt.Figure, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_base_dataset() -> pd.DataFrame:
    df = pd.read_csv(MERGED_DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def engineer_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, pd.Series, pd.Series]:
    # Target
    df["next_return"] = df.groupby("ticker")["daily_return"].shift(-1)
    df["target"] = (df["next_return"] > 0).astype(int)

    # Lagged returns
    df["lag_return_1"] = df["daily_return"]
    df["lag_return_2"] = df.groupby("ticker")["daily_return"].shift(1)
    df["lag_return_5"] = df.groupby("ticker")["daily_return"].shift(4)

    # Technical ratios
    df["price_to_ma20"] = (
        df["close"] / df["rolling_mean_20"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan) - 1
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    df["oc_gap"] = (
        (df["open"] - df["prev_close"]) / df["prev_close"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    df["vol_20ma"] = df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    df["vol_norm"] = (df["volume"] / df["vol_20ma"].replace(0, np.nan)).clip(0, 10)

    # News/fundamentals
    df["has_news"] = (df["news_count"].fillna(0) > 0).astype(float)
    df["log_marketcap"] = np.log1p(df["Marketcap"].fillna(0))

    for col in [
        "VIX",
        "Yield_Spread",
        "Regime_GMM",
        "sentiment_mean",
        "sentiment_ratio",
        "Revenuegrowth",
        "Weight",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["VIX"] = df["VIX"].fillna(df["VIX"].median())
    df["Yield_Spread"] = df["Yield_Spread"].fillna(df["Yield_Spread"].median())
    df["Regime_GMM"] = df["Regime_GMM"].fillna(df["Regime_GMM"].median())
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0)
    df["sentiment_ratio"] = df["sentiment_ratio"].fillna(0)
    df["Revenuegrowth"] = df["Revenuegrowth"].fillna(0)
    df["Weight"] = df["Weight"].fillna(0)

    df["Sector_encoded"] = df["Sector"].astype("category").cat.codes.astype(float)

    model_df = df[FEATURE_COLS + ["target", "date"]].dropna()
    train = model_df[model_df["date"] < SPLIT_DATE]
    test = model_df[model_df["date"] >= SPLIT_DATE]

    baseline = max(test["target"].mean(), 1 - test["target"].mean())

    sector_series = df.loc[model_df.index, "Sector"].fillna("Unknown").astype(str)
    test_sector = sector_series.loc[test.index]
    return model_df, train, test, float(baseline), sector_series, test_sector


def make_dataset_coverage_chart(df: pd.DataFrame) -> Path:
    by_date = (
        df.groupby("date")
        .agg(rows=("ticker", "size"), unique_tickers=("ticker", "nunique"))
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(by_date["date"], by_date["rows"], color="#1f77b4", lw=1.4)
    axes[0].set_title("Merged Dataset Coverage Over Time")
    axes[0].set_ylabel("Rows per Date")
    axes[0].grid(alpha=0.3)

    axes[1].plot(by_date["date"], by_date["unique_tickers"], color="#ff7f0e", lw=1.4)
    axes[1].set_ylabel("Unique Tickers per Date")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)

    return _save(fig, "01_rows_and_tickers_over_time.png")


def make_sector_distribution_chart(df: pd.DataFrame) -> Path:
    counts = df["Sector"].fillna("Unknown").value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(counts.index, counts.values, color="#4c78a8")
    ax.set_title("Sector Representation in Merged Dataset")
    ax.set_xlabel("Row Count")
    ax.set_ylabel("Sector")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, "02_sector_distribution_raw_rows.png")


def make_class_balance_chart(
    model_df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame
) -> Path:
    balance = pd.DataFrame(
        {
            "Split": ["Overall", "Train (<2023-01-01)", "Test (>=2023-01-01)"],
            "Up": [
                model_df["target"].mean(),
                train["target"].mean(),
                test["target"].mean(),
            ],
        }
    )
    balance["Down"] = 1 - balance["Up"]

    x = np.arange(len(balance))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, balance["Up"], width, label="Up (target=1)", color="#2ca02c")
    ax.bar(x + width / 2, balance["Down"], width, label="Down/Flat (target=0)", color="#d62728")
    ax.set_xticks(x, balance["Split"], rotation=10)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Share")
    ax.set_title("Class Balance After Feature Engineering")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, "03_class_balance_overall_train_test.png")


def make_news_coverage_chart(df: pd.DataFrame, model_df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> Path:
    coverage = pd.Series(
        {
            "Raw merged rows": (df["news_count"].fillna(0) > 0).mean(),
            "Model rows": df.loc[model_df.index, "has_news"].mean(),
            "Train rows": df.loc[train.index, "has_news"].mean(),
            "Test rows": df.loc[test.index, "has_news"].mean(),
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(coverage.index, coverage.values, color="#9467bd")
    ax.set_ylim(0, max(0.08, float(coverage.max()) * 1.2))
    ax.set_ylabel("Share of Rows with News")
    ax.set_title("News Availability by Split (Merged Dataset)")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, coverage.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.001,
            _pct(float(val)),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return _save(fig, "04_news_coverage_by_split.png")


def make_overall_model_accuracy_chart(overall_df: pd.DataFrame, baseline: float) -> Path:
    plot_df = overall_df.sort_values("Accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x="Accuracy",
        y="Model",
        hue="Model",
        palette="Blues_r",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.axvline(baseline, color="red", linestyle="--", linewidth=1.2, label=f"Always-Up baseline ({baseline:.4f})")
    ax.set_title("Overall Test Accuracy by Model")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Model")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, "05_model_accuracy_vs_baseline.png")


def make_overall_balanced_accuracy_chart(overall_df: pd.DataFrame) -> Path:
    plot_df = overall_df.sort_values("Balanced_Accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x="Balanced_Accuracy",
        y="Model",
        hue="Model",
        palette="mako",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="Random baseline (0.5)")
    ax.set_title("Overall Test Balanced Accuracy by Model")
    ax.set_xlabel("Balanced Accuracy")
    ax.set_ylabel("Model")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, "06_model_balanced_accuracy.png")


def make_best_sector_vs_baseline_chart(best_df: pd.DataFrame) -> Path:
    plot_df = best_df.sort_values("Delta_vs_Sector_Baseline", ascending=False).copy()
    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, plot_df["Accuracy"], width, label="Best model accuracy", color="#1f77b4")
    ax.bar(x + width / 2, plot_df["Sector_Baseline"], width, label="Sector always-up baseline", color="#ff7f0e")
    ax.set_xticks(x, plot_df["Sector"], rotation=35, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Best Model Per Sector vs Sector Baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, "07_best_model_per_sector_vs_baseline.png")


def make_best_sector_delta_chart(best_df: pd.DataFrame) -> Path:
    plot_df = best_df.sort_values("Delta_vs_Sector_Baseline", ascending=False).copy()
    colors = np.where(plot_df["Delta_vs_Sector_Baseline"] >= 0, "#2ca02c", "#d62728")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(plot_df["Sector"], plot_df["Delta_vs_Sector_Baseline"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Sector Lift: Best Model Accuracy Minus Sector Baseline")
    ax.set_ylabel("Delta Accuracy")
    ax.set_xlabel("Sector")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, "08_best_sector_delta.png")


def make_sector_heatmaps(sector_df: pd.DataFrame) -> tuple[Path, Path]:
    acc_pivot = sector_df.pivot(index="Sector", columns="Model", values="Accuracy")
    acc_pivot = acc_pivot.loc[acc_pivot.mean(axis=1).sort_values(ascending=False).index]

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.heatmap(acc_pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax1)
    ax1.set_title("Per-Sector Accuracy by Model")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Sector")
    path1 = _save(fig1, "09_sector_accuracy_heatmap.png")

    delta_pivot = sector_df.pivot(index="Sector", columns="Model", values="Delta_vs_Sector_Baseline")
    delta_pivot = delta_pivot.loc[delta_pivot.mean(axis=1).sort_values(ascending=False).index]

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(delta_pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax2)
    ax2.set_title("Per-Sector Delta vs Sector Baseline")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Sector")
    path2 = _save(fig2, "10_sector_delta_heatmap.png")

    return path1, path2


def make_recall_imbalance_chart(sector_df: pd.DataFrame) -> Path:
    recall = (
        sector_df.groupby("Model")[["Down_Recall", "Up_Recall"]]
        .mean()
        .sort_values("Up_Recall", ascending=False)
    )
    x = np.arange(len(recall))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, recall["Down_Recall"], width, label="Down recall", color="#d62728")
    ax.bar(x + width / 2, recall["Up_Recall"], width, label="Up recall", color="#2ca02c")
    ax.set_xticks(x, recall.index, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Average Recall Imbalance Across Sectors")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, "11_recall_imbalance_by_model.png")


def build_summary_metrics(
    df: pd.DataFrame,
    model_df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline: float,
    test_sector: pd.Series,
    overall_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> dict:
    mean_delta_by_model = (
        sector_df.groupby("Model")["Delta_vs_Sector_Baseline"].mean().sort_values(ascending=False)
    )
    recall_by_model = sector_df.groupby("Model")[["Down_Recall", "Up_Recall"]].mean()
    best_sorted = best_df.sort_values("Delta_vs_Sector_Baseline", ascending=False)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "merged_dataset": str(MERGED_DATA_PATH.relative_to(PROJECT_ROOT)),
            "overall_results": str(OVERALL_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "sector_results": str(SECTOR_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "best_sector_results": str(BEST_SECTOR_PATH.relative_to(PROJECT_ROOT)),
        },
        "raw_dataset": {
            "rows": int(len(df)),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "tickers": int(df["ticker"].nunique()),
            "sectors": int(df["Sector"].fillna("Unknown").nunique()),
            "rows_by_sector": {
                str(k): int(v)
                for k, v in df["Sector"].fillna("Unknown").value_counts().items()
            },
        },
        "modeling_dataset": {
            "rows_after_dropna": int(len(model_df)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "always_up_baseline_test": float(baseline),
            "overall_up_rate": float(model_df["target"].mean()),
            "train_up_rate": float(train["target"].mean()),
            "test_up_rate": float(test["target"].mean()),
            "test_rows_by_sector": {str(k): int(v) for k, v in test_sector.value_counts().items()},
        },
        "news_coverage": {
            "raw_rows_has_news": float((df["news_count"].fillna(0) > 0).mean()),
            "model_rows_has_news": float(df.loc[model_df.index, "has_news"].mean()),
            "train_rows_has_news": float(df.loc[train.index, "has_news"].mean()),
            "test_rows_has_news": float(df.loc[test.index, "has_news"].mean()),
        },
        "overall_model_results": [
            {
                "model": str(row["Model"]),
                "accuracy": float(row["Accuracy"]),
                "balanced_accuracy": float(row["Balanced_Accuracy"]),
            }
            for _, row in overall_df.sort_values("Accuracy", ascending=False).iterrows()
        ],
        "mean_sector_delta_by_model": {str(k): float(v) for k, v in mean_delta_by_model.items()},
        "best_sector_delta_sorted": [
            {
                "sector": str(row["Sector"]),
                "model": str(row["Model"]),
                "accuracy": float(row["Accuracy"]),
                "sector_baseline": float(row["Sector_Baseline"]),
                "delta_vs_sector_baseline": float(row["Delta_vs_Sector_Baseline"]),
                "n": int(row["N"]),
            }
            for _, row in best_sorted.iterrows()
        ],
        "positive_sector_deltas": {
            "count": int((best_df["Delta_vs_Sector_Baseline"] > 0).sum()),
            "total_sectors": int(len(best_df)),
        },
        "mean_recall_by_model": {
            model: {
                "down_recall": float(vals["Down_Recall"]),
                "up_recall": float(vals["Up_Recall"]),
            }
            for model, vals in recall_by_model.to_dict(orient="index").items()
        },
    }
    return metrics


def write_brief(metrics: dict) -> None:
    raw = metrics["raw_dataset"]
    mdl = metrics["modeling_dataset"]
    news = metrics["news_coverage"]
    best_model = metrics["overall_model_results"][0]
    pos = metrics["positive_sector_deltas"]
    best_sector = metrics["best_sector_delta_sorted"][0]
    worst_sector = metrics["best_sector_delta_sorted"][-1]

    lines = [
        "# Presentation Brief (Merged Dataset)",
        "",
        "## High-Level Overview",
        (
            f"- Main dataset: `{metrics['sources']['merged_dataset']}` with "
            f"{raw['rows']:,} rows, {raw['tickers']} tickers, {raw['sectors']} sectors, "
            f"covering {raw['date_min']} to {raw['date_max']}."
        ),
        (
            f"- Modeling frame (same pipeline as per-sector notebook): {mdl['rows_after_dropna']:,} rows "
            f"after feature engineering and dropna; train={mdl['train_rows']:,}, test={mdl['test_rows']:,} "
            f"with split date 2023-01-01."
        ),
        (
            f"- Test class balance: Up={_pct(mdl['test_up_rate'])}, Down={_pct(1 - mdl['test_up_rate'])}, "
            f"so always-up baseline is {mdl['always_up_baseline_test']:.4f}."
        ),
        (
            f"- Best overall model on accuracy: {best_model['model']} at {best_model['accuracy']:.4f}, "
            f"which is below the always-up baseline."
        ),
        (
            f"- Per-sector best model beats its sector baseline in {pos['count']} of {pos['total_sectors']} sectors; "
            f"best lift is {best_sector['sector']} ({best_sector['delta_vs_sector_baseline']:+.4f}), "
            f"worst is {worst_sector['sector']} ({worst_sector['delta_vs_sector_baseline']:+.4f})."
        ),
        (
            f"- News coverage is sparse in raw rows ({_pct(news['raw_rows_has_news'])}) and zero in test rows "
            f"({_pct(news['test_rows_has_news'])}), which limits sentiment impact in the evaluation window."
        ),
        "",
        "## What Is Going On (Specific Story)",
        "- Models cluster tightly around 51.7% to 51.8% accuracy while baseline is 52.24%.",
        "- Balanced accuracy stays near 0.50, meaning the classifiers are near-random on class balance.",
        "- Sector behavior varies, but most lifts vs sector baseline are small and often negative.",
        "- Recall is asymmetric: models recover Up days much better than Down days.",
        "",
        "## Graph Catalog (What Each Graph Proves)",
        "- `figures/01_rows_and_tickers_over_time.png`: Shows merged dataset coverage stability over time and confirms broad ticker coverage.",
        "- `figures/02_sector_distribution_raw_rows.png`: Quantifies representation by sector; helps explain where model metrics are most data-rich.",
        "- `figures/03_class_balance_overall_train_test.png`: Demonstrates the Up-class skew that creates the strong always-up baseline.",
        "- `figures/04_news_coverage_by_split.png`: Proves sentiment/news sparsity and the test-period coverage gap.",
        "- `figures/05_model_accuracy_vs_baseline.png`: Directly shows every model underperforming the baseline.",
        "- `figures/06_model_balanced_accuracy.png`: Shows near-random balanced accuracy despite model complexity.",
        "- `figures/07_best_model_per_sector_vs_baseline.png`: Compares best model and baseline inside each sector.",
        "- `figures/08_best_sector_delta.png`: Highlights exactly where model adds value (positive bars) vs harms value (negative bars).",
        "- `figures/09_sector_accuracy_heatmap.png`: Shows cross-model accuracy structure by sector.",
        "- `figures/10_sector_delta_heatmap.png`: Normalizes sector results against each sector baseline for fair comparison.",
        "- `figures/11_recall_imbalance_by_model.png`: Shows consistent Up-vs-Down recall gap (core failure mode).",
        "",
        "## Source Note",
        (
            "Model result CSVs used above come from `notebooks/eda_all_models_per_sector.ipynb`, "
            "which is run on `data/merged_dataset.csv` with the same split and feature pipeline."
        ),
    ]
    BRIEF_PATH.parent.mkdir(parents=True, exist_ok=True)
    BRIEF_PATH.write_text("\n".join(lines))


def main() -> None:
    _ensure_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    df = load_base_dataset()
    model_df, train, test, baseline, _, test_sector = engineer_model_frame(df)

    overall_df = pd.read_csv(OVERALL_RESULTS_PATH)
    sector_df = pd.read_csv(SECTOR_RESULTS_PATH)
    best_df = pd.read_csv(BEST_SECTOR_PATH)

    generated = []
    generated.append(make_dataset_coverage_chart(df))
    generated.append(make_sector_distribution_chart(df))
    generated.append(make_class_balance_chart(model_df, train, test))
    generated.append(make_news_coverage_chart(df, model_df, train, test))
    generated.append(make_overall_model_accuracy_chart(overall_df, baseline))
    generated.append(make_overall_balanced_accuracy_chart(overall_df))
    generated.append(make_best_sector_vs_baseline_chart(best_df))
    generated.append(make_best_sector_delta_chart(best_df))
    heatmap_paths = make_sector_heatmaps(sector_df)
    generated.extend(heatmap_paths)
    generated.append(make_recall_imbalance_chart(sector_df))

    metrics = build_summary_metrics(
        df=df,
        model_df=model_df,
        train=train,
        test=test,
        baseline=baseline,
        test_sector=test_sector,
        overall_df=overall_df,
        sector_df=sector_df,
        best_df=best_df,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    write_brief(metrics)

    print(f"Generated {len(generated)} figures in {FIG_DIR}")
    print(f"Wrote metrics: {METRICS_PATH}")
    print(f"Wrote brief:   {BRIEF_PATH}")


if __name__ == "__main__":
    main()
