# Presentation Brief (Merged Dataset)

## High-Level Overview
- Main dataset: `data/merged_dataset.csv` with 1,264,542 rows, 503 tickers, 12 sectors, covering 2016-02-10 to 2026-02-09.
- Modeling frame (same pipeline as per-sector notebook): 1,219,484 rows after feature engineering and dropna; train=830,418, test=389,066 with split date 2023-01-01.
- Test class balance: Up=52.24%, Down=47.76%, so always-up baseline is 0.5224.
- Best overall model on accuracy: Logistic Regression at 0.5182, which is below the always-up baseline.
- Per-sector best model beats its sector baseline in 6 of 12 sectors; best lift is Utilities (+0.0028), worst is Unknown (-0.0094).
- News coverage is sparse in raw rows (6.28%) and zero in test rows (0.00%), which limits sentiment impact in the evaluation window.

## What Is Going On (Specific Story)
- Models cluster tightly around 51.7% to 51.8% accuracy while baseline is 52.24%.
- Balanced accuracy stays near 0.50, meaning the classifiers are near-random on class balance.
- Sector behavior varies, but most lifts vs sector baseline are small and often negative.
- Recall is asymmetric: models recover Up days much better than Down days.

## Graph Catalog (What Each Graph Proves)
- `figures/01_rows_and_tickers_over_time.png`: Shows merged dataset coverage stability over time and confirms broad ticker coverage.
- `figures/02_sector_distribution_raw_rows.png`: Quantifies representation by sector; helps explain where model metrics are most data-rich.
- `figures/03_class_balance_overall_train_test.png`: Demonstrates the Up-class skew that creates the strong always-up baseline.
- `figures/04_news_coverage_by_split.png`: Proves sentiment/news sparsity and the test-period coverage gap.
- `figures/05_model_accuracy_vs_baseline.png`: Directly shows every model underperforming the baseline.
- `figures/06_model_balanced_accuracy.png`: Shows near-random balanced accuracy despite model complexity.
- `figures/07_best_model_per_sector_vs_baseline.png`: Compares best model and baseline inside each sector.
- `figures/08_best_sector_delta.png`: Highlights exactly where model adds value (positive bars) vs harms value (negative bars).
- `figures/09_sector_accuracy_heatmap.png`: Shows cross-model accuracy structure by sector.
- `figures/10_sector_delta_heatmap.png`: Normalizes sector results against each sector baseline for fair comparison.
- `figures/11_recall_imbalance_by_model.png`: Shows consistent Up-vs-Down recall gap (core failure mode).

## Source Note
Model result CSVs used above come from `notebooks/eda_all_models_per_sector.ipynb`, which is run on `data/merged_dataset.csv` with the same split and feature pipeline.