#!/usr/bin/env bash
# =============================================================================
# colombo-restaurant-pipeline/run.sh
#
# Usage:
#   ./run.sh --mock       Generate 150 mock restaurants (no API key needed)
#   ./run.sh --collect    Collect real data from Google Places API
#   ./run.sh --setup      Install Python dependencies only
# =============================================================================
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts"

# ── Shared venv (lives in backend project, shared across Python projects) ──────
VENV_DIR="$ROOT_DIR/colombo-restaurant-backend/venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"

# ── Logs ───────────────────────────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/pipeline.log"
[[ -f "$LOG_FILE" ]] && mv "$LOG_FILE" "${LOG_FILE%.log}.prev.log"

source "$SCRIPTS_DIR/common.sh"

# ── Flags ──────────────────────────────────────────────────────────────────────
DO_MOCK=false; DO_COLLECT=false; DO_SETUP=false; DO_VALIDATE=false

for arg in "$@"; do
  case "$arg" in
    --mock)     DO_MOCK=true ;;
    --collect)  DO_COLLECT=true ;;
    --setup)    DO_SETUP=true ;;
    --validate) DO_VALIDATE=true ;;
    --help|-h)
      echo "Usage: $0 [--mock] [--collect] [--setup] [--validate]"
      echo "  --mock      Generate 150 mock restaurants (no API key needed)"
      echo "  --collect   Collect real data from Google Places (needs GOOGLE_API_KEY in .env)"
      echo "  --setup     Install Python dependencies only"
      echo "  --validate  Load existing CSV into PySpark: schema check, null audit, stats, Parquet"
      exit 0 ;;
    *) warn "Unknown argument: $arg" ;;
  esac
done

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

# ── Install deps (if requested or venv fresh) ─────────────────────────────────
if [[ "$DO_SETUP" == "true" ]]; then
  step "Installing pipeline dependencies"
  run_with_log "$LOG_FILE" "pip (pipeline)" \
    "$VENV_PIP" install -r "$PROJECT_DIR/requirements.txt"
  success "Dependencies installed"
fi

# ── Generate / collect data ────────────────────────────────────────────────────
CSV_PATH="$PROJECT_DIR/colombo_restaurants.csv"

if [[ "$DO_COLLECT" == "true" ]]; then
  step "Collecting real restaurant data (Google Places)"
  [[ -z "${GOOGLE_API_KEY:-}" ]] && error "GOOGLE_API_KEY not set — add it to $PROJECT_DIR/.env"
  detail "This takes 5–15 min depending on API quota"
  detail "PySpark validation runs automatically after collection"
  run_with_log "$LOG_FILE" "pipeline.py" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' pipeline.py"
  success "Data collection + Spark validation complete  →  $CSV_PATH"

elif [[ "$DO_MOCK" == "true" ]]; then
  step "Generating mock restaurant data"
  detail "PySpark validation runs automatically after generation"
  run_with_log "$LOG_FILE" "generate_mock_data.py" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' generate_mock_data.py"
  success "Mock data + Spark validation complete  →  $CSV_PATH"

elif [[ "$DO_VALIDATE" == "true" ]]; then
  step "Running Spark validation on existing CSV"
  if [[ ! -f "$CSV_PATH" ]]; then
    error "CSV not found at $CSV_PATH — run --mock or --collect first"
  fi
  run_with_log "$LOG_FILE" "validate (standalone)" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' -c \"
from pipeline import load_into_spark
import os
load_into_spark(os.getenv('OUTPUT_FILE','colombo_restaurants.csv'), os.getenv('PARQUET_DIR', os.path.join('data','restaurants.parquet')))
\""
  success "Spark validation complete — see $LOG_FILE for details"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
if [[ -f "$CSV_PATH" ]]; then
  ROW_COUNT=$(( $(wc -l < "$CSV_PATH") - 1 ))
  PARQUET_DIR="$PROJECT_DIR/data/restaurants.parquet"
  success "Restaurant CSV ready  →  $ROW_COUNT restaurants  ($CSV_PATH)"
  [[ -d "$PARQUET_DIR" ]] && success "Parquet ready  →  $PARQUET_DIR"
else
  warn "No CSV yet — run with --mock or --collect to generate data"
fi
