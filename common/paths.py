"""Central path registry for the Colombo Restaurant pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Raw / processed data ──────────────────────────────────────
RAW_CSV          = str(ROOT / "data" / "raw"       / "colombo_restaurants_raw.csv")
CLEAN_CSV        = str(ROOT / "data" / "processed" / "colombo_restaurants_clean.csv")
ANALYTICS_DIR    = str(ROOT / "data" / "processed" / "analytics")
VIZ_DIR          = str(ROOT / "data" / "processed" / "visualizations")

# ── Final / model artefacts ───────────────────────────────────
FINAL_CSV        = str(ROOT / "data" / "final" / "colombo_restaurants_final.csv")
FEATURES_CSV     = str(ROOT / "data" / "final" / "colombo_restaurants_features.csv")
RECS_CSV         = str(ROOT / "data" / "final" / "all_user_recommendations.csv")
METRICS_JSON     = str(ROOT / "data" / "final" / "evaluation_metrics.json")
FEATURE_PIPELINE = str(ROOT / "models" / "feature_pipeline")
ALS_MODEL_DIR    = str(ROOT / "models" / "als_model")

# ── ALS hyper-parameters (single source of truth) ─────────────
ALS_RANK   = 10
ALS_ITER   = 10
ALS_REG    = 0.1
TRAIN_SEED = 42
TRAIN_SPLIT = [0.8, 0.2]
