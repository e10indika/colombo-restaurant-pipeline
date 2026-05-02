# Colombo Restaurant Data Pipeline

End-to-end Big Data Analytics and ALS Recommendation System for Colombo restaurants.  
Built with **Apache PySpark**, **Python 3.10+**, and **Google Places API**.

---

## Project Structure

```
colombo-restaurant-pipeline/
│
├── requirements.txt
├── run.sh                             # Unified run script (all flags below)
├── .env / .env.example
│
├── common/                            # Shared utilities (imported by part_a + part_b)
│   ├── spark_utils.py                 # build_spark() factory
│   ├── paths.py                       # all file path + ALS hyperparameter constants
│   ├── constants.py                   # PRICE_LABELS, RATING_BUCKETS, N_USERS, etc.
│   ├── io.py                          # load_csv() / save_csv() helpers
│   └── restaurant_index.py            # add_restaurant_index() (integer ID assignment)
│
├── part_a/                            # Big Data Analytics
│   ├── 01_data_collection.py          # Real data collector (Google Places API)
│   ├── 02_data_loading.py             # PySpark validation + Parquet persistence
│   ├── generate_mock_data.py          # Mock data generator (no API key needed)
│   ├── 03_data_cleaning.py
│   ├── 04_eda_analytics.py
│   └── 05_visualizations.py
│
├── part_b/                            # Recommendation System
│   ├── 01_load_clean_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_model_als.py
│   ├── 04_recommendations.py
│   └── 05_evaluation.py
│
├── data/
│   ├── raw/
│   │   └── colombo_restaurants_raw.csv      ← written by 01_data_collection / mock
│   ├── processed/
│   │   ├── colombo_restaurants_clean.csv    ← written by 03_data_cleaning
│   │   ├── analytics/                       ← 15 CSV files (one per analysis)
│   │   └── visualizations/                 ← 10 PNG charts
│   └── final/
│       ├── colombo_restaurants_final.csv    ← validated, recommendation-ready
│       ├── colombo_restaurants_features.csv ← ALS features (StringIndexer + scores)
│       ├── all_user_recommendations.csv     ← top-10 recs for every user
│       └── evaluation_metrics.json          ← RMSE, MAE, coverage, diversity
│
└── models/
    ├── feature_pipeline/                   ← fitted ML Pipeline (save/load)
    └── als_model/                          ← trained ALS model (save/load)
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Java | 11 or 17 (required by PySpark) |
| Node | — (frontend only) |

> **Note**: This pipeline shares the Python venv with `colombo-restaurant-backend/`.  
> Run the backend setup first if starting from scratch.

> **Note**: `run.sh` automatically sets `PYTHONPATH` to the pipeline root so the `common/` package is importable from every `part_a/` and `part_b/` script.

---

## Quick Start

### Step 0 — Install dependencies

```bash
# Install backend venv (shared)
cd ../colombo-restaurant-backend && ./run.sh --setup && cd -

# Install pipeline-specific deps into the shared venv
./run.sh --setup
```

### Step 1 — Generate data

```bash
# Option A: Mock data (no API key needed) — fastest for local dev
./run.sh --mock

# Option B: Real data from Google Places (requires API key in .env)
cp .env.example .env
# Edit .env → set GOOGLE_API_KEY=...
./run.sh --collect
```

### Step 2 — Run Part A Analytics

```bash
# Run all 3 Part A scripts in sequence
./run.sh --analytics

# Or run individually
./run.sh --clean    # data cleaning → data/processed/colombo_restaurants_clean.csv
./run.sh --eda      # 15 analyses   → data/processed/analytics/
./run.sh --viz      # 10 charts     → data/processed/visualizations/
```

### Step 3 — Run Part B Recommendation Model

```bash
# Run all 5 Part B scripts in sequence
./run.sh --model

# Or run individually
./run.sh --load        # validate + save final CSV
./run.sh --features    # feature engineering (StringIndexer, MinMaxScaler)
./run.sh --train-als   # train ALS model
./run.sh --recommend   # generate top-10 recommendations for all users
./run.sh --evaluate    # RMSE, MAE, coverage, diversity
```

### Full pipeline in one command

```bash
./run.sh --all
# Equivalent to: --mock --analytics --model
```

---

## Data Flow

```
part_a/01_data_collection.py  (Google Places API)
part_a/generate_mock_data.py  (mock — no API key)
           │
           ▼ part_a/02_data_loading.py (PySpark validation → Parquet)
 colombo_restaurants.csv  (root snapshot)
 data/raw/colombo_restaurants_raw.csv  (canonical raw for Part A)
           │
           ▼ part_a/03_data_cleaning.py
 data/processed/colombo_restaurants_clean.csv
           │
           ├──▶ part_a/04_eda_analytics.py  →  data/processed/analytics/ (15 CSVs)
           │
           ├──▶ part_a/05_visualizations.py →  data/processed/visualizations/ (10 PNGs)
           │
           └──▶ part_b/01_load_clean_data.py
                     │
                     ▼
           data/final/colombo_restaurants_final.csv
                     │
                     ▼ part_b/02_feature_engineering.py
           data/final/colombo_restaurants_features.csv
           models/feature_pipeline/
                     │
                     ▼ part_b/03_model_als.py
           models/als_model/
                     │
                     ├──▶ part_b/04_recommendations.py → data/final/all_user_recommendations.csv
                     └──▶ part_b/05_evaluation.py      → data/final/evaluation_metrics.json
```

---

## run.sh Flag Reference

| Flag | What it does |
|---|---|
| `--setup` | Install Python dependencies into the shared venv |
| `--mock` | Generate 150 mock restaurants; saves raw CSV + Parquet |
| `--collect` | Collect real data from Google Places (needs `GOOGLE_API_KEY`) |
| `--validate` | Load existing CSV into PySpark: schema check, null audit, stats |
| `--clean` | Run `part_a/03_data_cleaning.py` |
| `--eda` | Run `part_a/04_eda_analytics.py` |
| `--viz` | Run `part_a/05_visualizations.py` |
| `--analytics` | Run all Part A scripts (`--clean --eda --viz`) |
| `--load` | Run `part_b/01_load_clean_data.py` |
| `--features` | Run `part_b/02_feature_engineering.py` |
| `--train-als` | Run `part_b/03_model_als.py` (trains and saves ALS model) |
| `--recommend` | Run `part_b/04_recommendations.py` |
| `--evaluate` | Run `part_b/05_evaluation.py` |
| `--model` | Run all Part B scripts |
| `--all` | Run everything: mock → Part A → Part B |

---

## Part A Scripts

### `03_data_cleaning.py`
- Loads `data/raw/colombo_restaurants_raw.csv` with enforced schema
- Deduplicates on `place_id`
- Drops rows with both `rating` AND `total_ratings` null
- Imputes: median rating, mode price_level, 0 for counts, `""` for text
- Extracts Colombo district from address (regex: `Colombo \d{1,2}`)
- Maps Google Place types → `cuisine_category` (Cafe, Restaurant, Bar, Takeaway…)
- Adds `popularity_score = log1p(total_ratings) × rating`
- Adds `price_label` (Free / Budget / Moderate / Expensive / Luxury)
- Saves to `data/processed/colombo_restaurants_clean.csv`

### `04_eda_analytics.py`
Runs 15 analyses and saves each as a CSV:

| # | Analysis |
|---|---|
| 01 | Restaurant count by cuisine |
| 02 | Restaurant count by district |
| 03 | Avg rating by cuisine |
| 04 | Avg rating by district |
| 05 | Avg price level by cuisine |
| 06 | Price label distribution |
| 07 | Top 10 highest rated |
| 08 | Top 10 most reviewed |
| 09 | Correlation: total_ratings ↔ rating |
| 10 | Correlation: price_level ↔ rating |
| 11 | Open now vs closed |
| 12 | Avg popularity by district |
| 13 | Avg popularity by cuisine |
| 14 | Rating bucket distribution (Poor/Average/Good/Excellent) |
| 15 | Top 5 districts with most Luxury restaurants |

### `05_visualizations.py`
Generates 10 PNG charts using pandas + matplotlib + seaborn:

| File | Chart |
|---|---|
| `01_cuisine_count.png` | Bar — count by cuisine |
| `02_district_count.png` | Bar — count by district |
| `03_top10_most_reviewed.png` | Horizontal bar — top 10 reviewed |
| `04_rating_boxplot_by_cuisine.png` | Box plot — rating by cuisine |
| `05_scatter_ratings_vs_total.png` | Scatter — ratings vs total (by price) |
| `06_price_label_pie.png` | Pie — price label distribution |
| `07_heatmap_rating_district_cuisine.png` | Heatmap — avg rating by district × cuisine |
| `08_rating_histogram.png` | Histogram — overall rating distribution |
| `09_popularity_by_district.png` | Bar — avg popularity by district |
| `10_rating_bucket_countplot.png` | Count plot — rating buckets |

---

## Part B Scripts

### ALS Model Configuration

| Parameter | Value |
|---|---|
| `rank` | 10 |
| `maxIter` | 10 |
| `regParam` | 0.1 |
| `coldStartStrategy` | drop |
| `implicitPrefs` | False |
| Train/test split | 80/20 (seed=42) |

### Evaluation Metrics

After running `--evaluate`, check `data/final/evaluation_metrics.json`:
```json
{
  "rmse": 0.123,
  "mae":  0.091,
  "coverage_percent": 87.5,
  "avg_diversity": 4.2,
  "als_params": { "rank": 10, "maxIter": 10, "regParam": 0.1 }
}
```

---

## Logs

All output is captured to `logs/pipeline.log`.  
The previous run's log is preserved as `logs/pipeline.prev.log`.

```bash
# Watch live progress
tail -f logs/pipeline.log
```

---

## Environment Variables

Copy `.env.example` to `.env` and set values:

```env
GOOGLE_API_KEY=your_key_here        # required for --collect
OUTPUT_FILE=colombo_restaurants.csv # optional: root snapshot path
HISTORICAL_FILE=data/historical_ratings.csv  # optional: delta CSV
PARQUET_DIR=data/restaurants.parquet         # optional: Parquet output
```
