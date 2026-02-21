# Predicting Short-Term Stock Price Direction

Final Data Mining project (Winter 25-26).

This repo builds a merged S&P 500 dataset, engineers technical/macro/sentiment features, and benchmarks short-term direction models (next-day up/down).

## Repository Highlights

- `data/merged_dataset.csv`: main modeling dataset (43 columns).
- `notebooks/`: EDA, baselines, and model notebooks.
- `notebooks/eda_all_models.ipynb`: unified model benchmark (same features/split).
- `notebooks/eda_all_models_per_sector.ipynb`: same benchmark + per-sector analysis.
- `scripts/`: data collection/feature generation/merge scripts.
- `generate_report.py`: generates `draft_report.docx`.

## Prerequisites

- Python 3.10+ (3.11 recommended)
- macOS/Linux shell commands below assume `bash`/`zsh`

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (only needed for `scripts/sentiment_analysis.py`):

```bash
pip install -r requirements-sentiment.txt
```

## Quick Start (Existing Data)

If `data/merged_dataset.csv` already exists, you can go straight to notebooks or report generation.

### Open notebooks

```bash
jupyter notebook
```

Recommended notebooks:

- `notebooks/eda_all_models.ipynb`
- `notebooks/eda_all_models_per_sector.ipynb`

### Generate report

```bash
python generate_report.py
```

Output:

- `draft_report.docx`

## Rebuild Data Pipeline (Optional)

The full upstream pipeline depends on external downloads/datasets. Typical order:

1. Build base S&P500 data:

```bash
python scripts/main.py
```

2. Build rolling features:

```bash
python scripts/rolling_data.py
```

3. Merge team datasets:

```bash
python scripts/merge_all.py
```

4. (Optional) Build sentiment aggregates:

```bash
python scripts/sentiment_analysis.py
```

## Model/Results Outputs

Per-sector notebook/export writes:

- `data/model_overall_results.csv`
- `data/model_sector_results.csv`
- `data/model_best_by_sector.csv`

## Troubleshooting

If report generation fails with an old `docx` package error (`No module named 'exceptions'`):

```bash
pip uninstall -y docx
pip install python-docx
```
