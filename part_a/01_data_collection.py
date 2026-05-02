"""
part_a/01_data_collection.py
==============================
Step 01 — Data Collection via Google Places API.

Collects Colombo restaurant data, enriches with synthetic busyness scores,
appends to a historical CSV for time-series analysis, and hands off to
02_data_loading.py for PySpark validation and Parquet persistence.

Usage
-----
    python part_a/01_data_collection.py

Environment variables (set in project root .env)
-------------------------------------------------
    GOOGLE_API_KEY   (required) Google Places API key
    OUTPUT_FILE      (optional) root snapshot CSV  [colombo_restaurants.csv]
    HISTORICAL_FILE  (optional) delta CSV          [data/historical_ratings.csv]
    PARQUET_DIR      (optional) Parquet output dir [data/restaurants.parquet]
"""

import hashlib
import importlib.util
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import requests
from tqdm import tqdm

# Project root is two levels up from part_a/
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')   # load from project root, not part_a/

# ── Import 02_data_loading by file path (digit prefix prevents standard import)
def _import_02():
    spec = importlib.util.spec_from_file_location(
        "data_loading",
        Path(__file__).resolve().parent / "02_data_loading.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

load_into_spark = _import_02().load_into_spark


# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
API_KEY = os.getenv("GOOGLE_API_KEY")          # Never hardcode credentials
if not API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set. "
        "Export it before running: export GOOGLE_API_KEY='your_key'"
    )

OUTPUT_FILE      = os.getenv("OUTPUT_FILE",     str(ROOT / "colombo_restaurants.csv"))
HISTORICAL_FILE  = os.getenv("HISTORICAL_FILE", str(ROOT / "data" / "historical_ratings.csv"))
PARQUET_DIR      = os.getenv("PARQUET_DIR",     str(ROOT / "data" / "restaurants.parquet"))
SEARCH_RADIUS    = 3000   # metres per location

# Colombo districts/areas
SEARCH_LOCATIONS = [
    {"name": "Colombo Fort",                 "lat": 6.9344, "lng": 79.8428},
    {"name": "Colombo 3 (Kollupitiya)",      "lat": 6.9102, "lng": 79.8497},
    {"name": "Colombo 4 (Bambalapitiya)",    "lat": 6.8952, "lng": 79.8553},
    {"name": "Colombo 5 (Havelock Town)",    "lat": 6.8870, "lng": 79.8651},
    {"name": "Colombo 6 (Wellawatte)",       "lat": 6.8756, "lng": 79.8609},
    {"name": "Colombo 7 (Cinnamon Gardens)", "lat": 6.9018, "lng": 79.8657},
    {"name": "Pettah",                       "lat": 6.9388, "lng": 79.8519},
    {"name": "Borella",                      "lat": 6.9181, "lng": 79.8774},
    {"name": "Nugegoda",                     "lat": 6.8728, "lng": 79.8894},
    {"name": "Dehiwala",                     "lat": 6.8519, "lng": 79.8653},
]

# ── Busyness model ─────────────────────────────────────────────────────────────

# Hourly weight curve (index = hour 0-23, Sri Lanka local time).
# Peaks at lunch (12–14) and dinner (18–21); quiet overnight.
_HOUR_WEIGHTS = [
    0.05, 0.03, 0.02, 0.02, 0.02, 0.04,   # 00–05
    0.10, 0.20, 0.30, 0.40, 0.55, 0.70,   # 06–11
    0.95, 1.00, 0.90, 0.75, 0.65, 0.70,   # 12–17
    0.90, 1.00, 0.95, 0.80, 0.60, 0.30,   # 18–23
]


def calculate_busyness_score(place_id: str, user_ratings_total: int | None) -> float:
    """
    Generate a synthetic 'live' busyness percentage (0–100 %) for a place.

    The score combines two signals:
      1. **Popularity baseline** — derived from `user_ratings_total` via a
         log-scale normalisation against a reference of 5 000 reviews
         (≈ top-tier Colombo restaurant).  More reviews → higher baseline.
      2. **Time-of-day multiplier** — a realistic hourly curve that peaks
         at meal times (12–14 h and 18–21 h) and troughs overnight.
      3. **Deterministic jitter** — a small hash-based offset tied to
         `place_id` so different restaurants vary even at the same hour,
         simulating individual popularity variance.

    Parameters
    ----------
    place_id : str
        Google Places ID (used only for deterministic jitter, not for an API call).
    user_ratings_total : int | None
        Total review count from the Places API.

    Returns
    -------
    float
        Busyness percentage in [0.0, 100.0], rounded to 1 decimal place.
    """
    total = user_ratings_total or 0
    current_hour = datetime.now(tz=timezone.utc).hour   # UTC; adjust if desired

    # 1. Popularity baseline: log-normalised to [0, 1]
    reference_reviews = 5_000
    popularity = math.log1p(total) / math.log1p(reference_reviews)
    popularity = min(popularity, 1.0)

    # 2. Time-of-day multiplier
    time_weight = _HOUR_WEIGHTS[current_hour % 24]

    # 3. Deterministic per-place jitter in [-0.07, +0.07]
    digest = int(hashlib.md5(place_id.encode()).hexdigest()[:8], 16)
    jitter = (digest % 15 - 7) / 100.0   # maps to [-0.07, +0.07]

    raw = (0.6 * popularity + 0.4 * time_weight) + jitter
    score = max(0.0, min(100.0, raw * 100.0))
    return round(score, 1)


# ── Step 1: Nearby Search ──────────────────────────────────────────────────────

def search_restaurants_nearby(
    lat: float, lng: float, radius: int = SEARCH_RADIUS, page_token: str | None = None
) -> dict:
    """Call the Places Nearby Search endpoint and return the raw JSON response."""
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "type": "restaurant",
        "key": API_KEY,
    }
    if page_token:
        params["pagetoken"] = page_token

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


# ── Step 2: Place Details ──────────────────────────────────────────────────────

def get_place_details(place_id: str) -> dict:
    """Fetch full details for a single place and return the result dict."""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": (
            "name,place_id,formatted_address,geometry,"
            "rating,user_ratings_total,price_level,"
            "types,opening_hours,website,formatted_phone_number,"
            "reviews"
        ),
        "key": API_KEY,
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json().get("result", {})


# ── Step 3: Collect All Place IDs ─────────────────────────────────────────────

def collect_place_ids() -> list[str]:
    """
    Iterate over all search locations (up to 3 pages each) and return a
    deduplicated list of Place IDs.
    """
    all_place_ids: set[str] = set()

    for area in SEARCH_LOCATIONS:
        logger.info("Searching: %s", area["name"])
        page_token = None

        for page in range(3):      # Google caps at 3 pages (≈ 60 results) per query
            if page > 0:
                time.sleep(2)      # mandatory delay before using next_page_token

            data = search_restaurants_nearby(area["lat"], area["lng"], page_token=page_token)
            results = data.get("results", [])

            for place in results:
                all_place_ids.add(place["place_id"])

            logger.debug("  page %d: %d results  (running total: %d)",
                         page + 1, len(results), len(all_place_ids))

            page_token = data.get("next_page_token")
            if not page_token:
                break

    logger.info("Total unique place IDs collected: %d", len(all_place_ids))
    return list(all_place_ids)


# ── Step 4: Fetch Details + Build Records ─────────────────────────────────────

def fetch_all_details(place_ids: list[str]) -> list[dict]:
    """
    Retrieve full details for each place ID, enrich with a busyness score,
    and return a list of flat record dicts ready for DataFrame conversion.
    """
    records: list[dict] = []
    collected_at = datetime.now(tz=timezone.utc).isoformat()

    for place_id in tqdm(place_ids, desc="Fetching details", unit="place"):
        details = get_place_details(place_id)
        if not details:
            logger.warning("Empty details for place_id=%s — skipping", place_id)
            continue

        reviews = details.get("reviews", [])
        review_texts = " | ".join(r.get("text", "") for r in reviews)
        avg_review_rating = (
            sum(r.get("rating", 0) for r in reviews) / len(reviews)
            if reviews else None
        )

        total_ratings = details.get("user_ratings_total")
        pid = details.get("place_id", place_id)

        record = {
            # ── Identity ──────────────────────────────────────────────────────
            "place_id":           pid,
            "name":               details.get("name"),
            "address":            details.get("formatted_address"),
            "lat":                details.get("geometry", {}).get("location", {}).get("lat"),
            "lng":                details.get("geometry", {}).get("location", {}).get("lng"),
            # ── Ratings ───────────────────────────────────────────────────────
            "rating":             details.get("rating"),
            "total_ratings":      total_ratings,
            "price_level":        details.get("price_level"),      # 0–4 scale
            # ── Busyness (synthetic) ──────────────────────────────────────────
            "busyness_score":     calculate_busyness_score(pid, total_ratings),
            # ── Metadata ──────────────────────────────────────────────────────
            "types":              ", ".join(details.get("types", [])),
            "phone":              details.get("formatted_phone_number"),
            "website":            details.get("website"),
            "open_now":           details.get("opening_hours", {}).get("open_now"),
            # ── Reviews ───────────────────────────────────────────────────────
            "review_count":       len(reviews),
            "review_texts":       review_texts,
            "avg_review_rating":  avg_review_rating,
            # ── Pipeline audit column ─────────────────────────────────────────
            "collected_at":       collected_at,
        }
        records.append(record)
        time.sleep(0.1)   # stay within API rate limits

    return records


# ── Step 5: Delta / Append to Historical CSV ──────────────────────────────────

def append_to_historical(df: pd.DataFrame, historical_path: str) -> None:
    """
    Append new records to the historical CSV without overwriting existing data.

    Strategy
    --------
    - If the file does not exist yet, write with a header row.
    - If the file already exists, append without writing the header again.
    - This produces a growing time-series file where each row carries a
      `collected_at` timestamp, enabling hourly trend / delta analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The freshly fetched records for this pipeline run.
    historical_path : str
        Path to the append-only historical CSV file.
    """
    os.makedirs(os.path.dirname(historical_path) or ".", exist_ok=True)

    file_exists = os.path.isfile(historical_path)
    df.to_csv(
        historical_path,
        mode   = "a",               # append — never truncate
        header = not file_exists,   # write header only on first creation
        index  = False,
    )

    action = "Updated" if file_exists else "Created"
    logger.info("%s historical file: %s  (+%d rows)", action, historical_path, len(df))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Colombo Restaurant Data Pipeline ===")

    # 1. Discover all unique restaurants
    place_ids = collect_place_ids()

    # 2. Fetch full details + busyness scores
    logger.info("Fetching details for %d places …", len(place_ids))
    records = fetch_all_details(place_ids)

    if not records:
        logger.error("No records collected — aborting.")
        return

    # 3. Build DataFrame and deduplicate within this run
    df = pd.DataFrame(records)
    df.drop_duplicates(subset="place_id", inplace=True)
    logger.info("Records after deduplication: %d", len(df))

    # 4. Write one-shot snapshot CSV (overwrites each run)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Snapshot written → %s", OUTPUT_FILE)

    # 4b. Write canonical raw CSV for Part A cleaning step
    raw_path = ROOT / "data" / "raw" / "colombo_restaurants_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(raw_path), index=False)
    logger.info("Raw data copy written → %s", raw_path)

    # 5. Append to historical CSV (never overwrites — delta / time-series)
    append_to_historical(df, HISTORICAL_FILE)

    # 6. Load into PySpark: validate schema, audit nulls, compute stats, save Parquet
    load_into_spark(OUTPUT_FILE, PARQUET_DIR)

    logger.info("=== Pipeline complete: %d restaurants ===", len(df))
    print(df[["name", "rating", "total_ratings", "busyness_score", "collected_at"]].head(10))


if __name__ == "__main__":
    main()
