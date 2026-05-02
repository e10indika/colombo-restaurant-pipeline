#!/usr/bin/env bash
# =============================================================================
# colombo-restaurant-pipeline/run.sh
#
# Data collection
#   ./run.sh --mock          Generate 150 mock restaurants (no API key needed)
#   ./run.sh --collect       Collect real data from Google Places API
#   ./run.sh --validate      Load existing CSV into PySpark: schema, null audit, Parquet
#   ./run.sh --setup         Install Python dependencies only
#
# Analytics pipeline (Part A)
#   ./run.sh --clean         Run 03_data_cleaning.py
#   ./run.sh --eda           Run 04_eda_analytics.py
#   ./run.sh --viz           Run 05_visualizations.py
#   ./run.sh --analytics     Run all Part A scripts (clean → eda → viz)
#
# Recommendation model (Part B)
#   ./run.sh --load          Run part_b/01_load_clean_data.py
#   ./run.sh --features      Run part_b/02_feature_engineering.py
#   ./run.sh --train-als     Run part_b/03_model_als.py
#   ./run.sh --recommend     Run part_b/04_recommendations.py
#   ./run.sh --evaluate      Run part_b/05_evaluation.py
#   ./run.sh --model         Run all Part B scripts (load → features → train → recommend → evaluate)
#
# Full pipeline
#   ./run.sh --all           mock → clean → eda → viz → load → features → train → recommend → evaluate
# =============================================================================
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts"

# ── Shared venv (lives in backend project, shared across Python projects) ──────
VENV_DIR="$ROOT_DIR/colombo-restaurant-backend/venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"

# ── Make the pipeline root importable so `from common.x import y` works ────────
# Python adds the script's own directory (e.g. part_a/) to sys.path, not the
# project root.  PYTHONPATH ensures the common/ package is always resolvable.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ── Logs ───────────────────────────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/pipeline.log"
[[ -f "$LOG_FILE" ]] && mv "$LOG_FILE" "${LOG_FILE%.log}.prev.log"

source "$SCRIPTS_DIR/common.sh"

# ── Flags ──────────────────────────────────────────────────────────────────────
DO_MOCK=false; DO_COLLECT=false; DO_SETUP=false; DO_VALIDATE=false
DO_CLEAN=false; DO_EDA=false; DO_VIZ=false; DO_ANALYTICS=false
DO_LOAD=false; DO_FEATURES=false; DO_TRAIN=false; DO_RECOMMEND=false; DO_EVALUATE=false
DO_MODEL=false; DO_ALL=false

for arg in "$@"; do
  case "$arg" in
    --mock)        DO_MOCK=true ;;
    --collect)     DO_COLLECT=true ;;
    --setup)       DO_SETUP=true ;;
    --validate)    DO_VALIDATE=true ;;
    --clean)       DO_CLEAN=true ;;
    --eda)         DO_EDA=true ;;
    --viz)         DO_VIZ=true ;;
    --analytics)   DO_ANALYTICS=true ;;
    --load)        DO_LOAD=true ;;
    --features)    DO_FEATURES=true ;;
    --train-als)   DO_TRAIN=true ;;
    --recommend)   DO_RECOMMEND=true ;;
    --evaluate)    DO_EVALUATE=true ;;
    --model)       DO_MODEL=true ;;
    --all)         DO_ALL=true ;;
    --help|-h)
      grep "^#   \./run" "$0" | sed 's/^# //'
      exit 0 ;;
    *) warn "Unknown argument: $arg" ;;
  esac
done

# --analytics expands to all Part A steps
[[ "$DO_ANALYTICS" == "true" ]] && DO_CLEAN=true && DO_EDA=true && DO_VIZ=true

# --model expands to all Part B steps
[[ "$DO_MODEL" == "true" ]] && DO_LOAD=true && DO_FEATURES=true && DO_TRAIN=true && DO_RECOMMEND=true && DO_EVALUATE=true

# --all expands to mock + all Part A + all Part B
if [[ "$DO_ALL" == "true" ]]; then
  DO_MOCK=true
  DO_CLEAN=true; DO_EDA=true; DO_VIZ=true
  DO_LOAD=true; DO_FEATURES=true; DO_TRAIN=true; DO_RECOMMEND=true; DO_EVALUATE=true
fi

echo ""
echo -e "\033[1m🗄️  Pipeline — Colombo Restaurant Data\033[0m"
echo "  Log → $LOG_FILE"
echo "============================================="
{ echo "===== Pipeline run $(date) ====="; } >> "$LOG_FILE"

# ── Load .env ──────────────────────────────────────────────────────────────────
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a; source "$PROJECT_DIR/.env"; set +a
  detail ".env loaded"
else
  warn "No .env found — copy .env.example and set GOOGLE_API_KEY"
fi

# ── Verify venv exists ────────────────────────────────────────────────────────
step "Checking Python environment"
if [[ ! -f "$VENV_PYTHON" ]]; then
  error "Shared venv not found at $VENV_DIR — run colombo-restaurant-backend/run.sh --setup first"
fi
success "Using shared venv  →  $($VENV_PYTHON --version)  ($VENV_PYTHON)"

# ── Install deps ──────────────────────────────────────────────────────────────
if [[ "$DO_SETUP" == "true" ]]; then
  step "Installing pipeline dependencies"
  run_with_log "$LOG_FILE" "pip (pipeline)" \
    "$VENV_PIP" install -r "$PROJECT_DIR/requirements.txt"
  success "Dependencies installed"
fi

# ── Ensure data directories exist ─────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/data/raw" \
         "$PROJECT_DIR/data/processed/analytics" \
         "$PROJECT_DIR/data/processed/visualizations" \
         "$PROJECT_DIR/data/final" \
         "$PROJECT_DIR/models"

CSV_PATH="$PROJECT_DIR/colombo_restaurants.csv"
RAW_CSV="$PROJECT_DIR/data/raw/colombo_restaurants_raw.csv"

# ══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$DO_COLLECT" == "true" ]]; then
  step "Collecting real restaurant data (Google Places)"
  [[ -z "${GOOGLE_API_KEY:-}" ]] && error "GOOGLE_API_KEY not set — add it to $PROJECT_DIR/.env"
  detail "This takes 5–15 min depending on API quota"
  run_with_log "$LOG_FILE" "01_data_collection" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' part_a/01_data_collection.py"
  success "Data collection complete  →  $CSV_PATH"

elif [[ "$DO_MOCK" == "true" ]]; then
  step "Generating mock restaurant data (150 restaurants)"
  run_with_log "$LOG_FILE" "generate_mock_data" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' part_a/generate_mock_data.py"
  success "Mock data ready  →  $CSV_PATH"
fi

if [[ "$DO_VALIDATE" == "true" ]]; then
  step "Running Spark validation on existing CSV"
  if [[ ! -f "$CSV_PATH" ]]; then
    error "CSV not found at $CSV_PATH — run --mock or --collect first"
  fi
  run_with_log "$LOG_FILE" "02_data_loading (validate)" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' part_a/02_data_loading.py"
  success "Spark validation complete — see $LOG_FILE"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PART A — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$DO_CLEAN" == "true" ]]; then
  step "Part A — Step 03: Data Cleaning"
  if [[ ! -f "$RAW_CSV" ]]; then
    error "Raw CSV not found at $RAW_CSV — run --mock or --collect first"
  fi
  run_with_log "$LOG_FILE" "03_data_cleaning" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' part_a/03_data_cleaning.py"
  success "Cleaned data → data/processed/colombo_restaurants_clean.csv"
fi

if [[ "$DO_EDA" == "true" ]]; then
  step "Part A — Step 04: EDA Analytics"
  if [[ ! -e "$PROJECT_DIR/data/processed/colombo_restaurants_clean.csv" ]]; then
    error "Clean CSV not found — run --clean first"
  fi
  run_with_log "$LOG_FILE" "04_eda_analytics" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' part_a/04_eda_analytics.py"
  success "Analytics CSVs → data/processed/analytics/"
fi

if [[ "$DO_VIZ" == "true" ]]; then
  step "Part A — Step 05: Visualizations"
  if [[ ! -e "$PROJECT_DIR/data/processed/colombo_restaurants_clean.csv" ]]; then
    error "Clean CSV not found — run --clean first"
  fi
  run_with_log "$LOG_FILE" "05_visualizations" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' part_a/05_visualizations.py"
  success "Charts (10 PNGs) → data/processed/visualizations/"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PART B — RECOMMENDATION MODEL
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$DO_LOAD" == "true" ]]; then
  step "Part B — Step 01: Load & Validate Clean Data"
  run_with_log "$LOG_FILE" "b01_load_clean_data" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' part_b/01_load_clean_data.py"
  success "Final dataset → data/final/colombo_restaurants_final.csv"
fi

if [[ "$DO_FEATURES" == "true" ]]; then
  step "Part B — Step 02: Feature Engineering"
  run_with_log "$LOG_FILE" "b02_feature_engineering" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' part_b/02_feature_engineering.py"
  success "Features → data/final/colombo_restaurants_features.csv + models/feature_pipeline"
fi

if [[ "$DO_TRAIN" == "true" ]]; then
  step "Part B — Step 03: ALS Model Training"
  run_with_log "$LOG_FILE" "b03_model_als" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' part_b/03_model_als.py"
  success "ALS model saved → models/als_model/"
fi

if [[ "$DO_RECOMMEND" == "true" ]]; then
  step "Part B — Step 04: Generate Recommendations"
  run_with_log "$LOG_FILE" "b04_recommendations" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' part_b/04_recommendations.py"
  success "Recommendations → data/final/all_user_recommendations.csv"
fi

if [[ "$DO_EVALUATE" == "true" ]]; then
  step "Part B — Step 05: Model Evaluation"
  run_with_log "$LOG_FILE" "b05_evaluation" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' part_b/05_evaluation.py"
  success "Metrics → data/final/evaluation_metrics.json"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "  📂 Output locations:"
[[ -f "$CSV_PATH" ]] && {
  ROW_COUNT=$(( $(wc -l < "$CSV_PATH") - 1 ))
  echo "     Raw CSV           $ROW_COUNT rows  →  $CSV_PATH"
}
[[ -e "$PROJECT_DIR/data/processed/colombo_restaurants_clean.csv" ]] && \
  echo "     Cleaned CSV                 →  data/processed/colombo_restaurants_clean.csv"
[[ -d "$PROJECT_DIR/data/processed/analytics" ]] && \
  echo "     Analytics CSVs              →  data/processed/analytics/"
[[ -d "$PROJECT_DIR/data/processed/visualizations" ]] && \
  echo "     Visualizations              →  data/processed/visualizations/"
[[ -e "$PROJECT_DIR/data/final/all_user_recommendations.csv" ]] && \
  echo "     Recommendations             →  data/final/all_user_recommendations.csv"
[[ -f "$PROJECT_DIR/data/final/evaluation_metrics.json" ]] && \
  echo "     Evaluation metrics          →  data/final/evaluation_metrics.json"
echo "     Log                         →  $LOG_FILE"

