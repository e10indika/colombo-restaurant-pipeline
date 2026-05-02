"""
Central path registry for the Colombo Restaurant pipeline.

Every path defaults to a __file__-relative location for local development.
Override any of them via environment variables for deployment on remote servers
where the repository layout may differ.

Environment variables
---------------------
    PIPELINE_RAW_CSV          RAW_CSV
    PIPELINE_CLEAN_CSV        CLEAN_CSV
    PIPELINE_ANALYTICS_DIR    ANALYTICS_DIR
    PIPELINE_VIZ_DIR          VIZ_DIR
    PIPELINE_FINAL_CSV        FINAL_CSV
    PIPELINE_FEATURES_CSV     FEATURES_CSV
    PIPELINE_RECS_CSV         RECS_CSV
    PIPELINE_METRICS_JSON     METRICS_JSON
    PIPELINE_FEATURE_PIPELINE FEATURE_PIPELINE
    PIPELINE_ALS_MODEL_DIR    ALS_MODEL_DIR
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _p(env_key: str, *rel_parts: str) -> str:
    """Return env var value if set, otherwise ROOT-relative path."""
    return os.getenv(env_key, str(ROOT.joinpath(*rel_parts)))


# ── Raw / processed data ──────────────────────────────────────────────────────
RAW_CSV       = _p('PIPELINE_RAW_CSV',       'data', 'raw',       'colombo_restaurants_raw.csv')
CLEAN_CSV     = _p('PIPELINE_CLEAN_CSV',     'data', 'processed', 'colombo_restaurants_clean.csv')
ANALYTICS_DIR = _p('PIPELINE_ANALYTICS_DIR', 'data', 'processed', 'analytics')
VIZ_DIR       = _p('PIPELINE_VIZ_DIR',       'data', 'processed', 'visualizations')

# ── Final / model artefacts ───────────────────────────────────────────────────
FINAL_CSV        = _p('PIPELINE_FINAL_CSV',        'data',   'final',  'colombo_restaurants_final.csv')
FEATURES_CSV     = _p('PIPELINE_FEATURES_CSV',     'data',   'final',  'colombo_restaurants_features.csv')
RECS_CSV         = _p('PIPELINE_RECS_CSV',         'data',   'final',  'all_user_recommendations.csv')
METRICS_JSON     = _p('PIPELINE_METRICS_JSON',     'data',   'final',  'evaluation_metrics.json')
FEATURE_PIPELINE = _p('PIPELINE_FEATURE_PIPELINE', 'models', 'feature_pipeline')
ALS_MODEL_DIR    = _p('PIPELINE_ALS_MODEL_DIR',    'models', 'als_model')

# ── ALS hyper-parameters (single source of truth for the pipeline) ────────────
ALS_RANK   = int(os.getenv('ALS_RANK',    '10'))
ALS_ITER   = int(os.getenv('ALS_ITER',    '10'))
ALS_REG    = float(os.getenv('ALS_REG',   '0.1'))
TRAIN_SEED = int(os.getenv('TRAIN_SEED',  '42'))
TRAIN_SPLIT = [0.8, 0.2]
