"""
part_a/generate_mock_data.py
==============================
Generates 150 realistic mock restaurants for local development
without requiring a Google Places API key.

Produces the same schema as 01_data_collection.py, then calls
02_data_loading.py for PySpark validation and Parquet output.

Usage
-----
    python part_a/generate_mock_data.py

Outputs
-------
    colombo_restaurants.csv              root snapshot (overwritten each run)
    data/raw/colombo_restaurants_raw.csv canonical raw CSV for 03_data_cleaning.py
    data/restaurants.parquet/            Parquet copy
"""

import csv
import hashlib
import importlib.util
import math
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

random.seed(42)

ROOT        = Path(__file__).resolve().parent.parent
OUTPUT_FILE = os.getenv("OUTPUT_FILE", str(ROOT / "colombo_restaurants.csv"))
PARQUET_DIR = os.getenv("PARQUET_DIR", str(ROOT / "data" / "restaurants.parquet"))

# Import load_into_spark from 02_data_loading.py by file path
def _import_02():
    spec = importlib.util.spec_from_file_location(
        "data_loading",
        Path(__file__).resolve().parent / "02_data_loading.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

load_into_spark = _import_02().load_into_spark

# ── Mock restaurant data ───────────────────────────────────────────────────────

AREAS = [
    ("Colombo 01", 6.9344, 79.8428),
    ("Colombo 02", 6.9200, 79.8480),
    ("Colombo 03", 6.9102, 79.8497),
    ("Colombo 04", 6.8952, 79.8553),
    ("Colombo 05", 6.8870, 79.8651),
    ("Colombo 06", 6.8756, 79.8609),
    ("Colombo 07", 6.9018, 79.8657),
    ("Pettah",     6.9388, 79.8519),
    ("Borella",    6.9181, 79.8774),
    ("Nugegoda",   6.8728, 79.8894),
    ("Dehiwala",   6.8519, 79.8653),
]

CUISINES = [
    ("seafood_restaurant",    "Seafood"),
    ("indian_restaurant",     "Indian"),
    ("chinese_restaurant",    "Chinese"),
    ("italian_restaurant",    "Italian"),
    ("fast_food_restaurant",  "Fast Food"),
    ("cafe",                  "Café"),
    ("bakery",                "Bakery"),
    ("buffet_restaurant",     "Buffet"),
    ("vegetarian_restaurant", "Vegetarian"),
    ("pizza_restaurant",      "Pizza"),
    ("thai_restaurant",       "Thai"),
    ("japanese_restaurant",   "Japanese"),
]

NAME_PREFIXES   = ["The", "Colombo", "Royal", "Ceylon", "Spice", "Golden", "Blue",
                   "Green", "Bay", "Garden", "Lake", "Pearl", "Ocean"]
NAME_SUFFIXES   = ["Kitchen", "Bistro", "Grill", "House", "Garden", "Café",
                   "Restaurant", "Lounge", "Dining", "Table", "Corner", "Palace"]
STREET_NAMES    = ["Galle Road", "Union Place", "Duplication Road", "High Level Road",
                   "Baseline Road", "Marine Drive", "D.S. Senanayake Mawatha",
                   "Bauddhaloka Mawatha", "R.A. De Mel Mawatha", "Havelock Road"]
PHONE_PREFIXES  = ["011", "077", "076", "071", "072"]

REVIEW_SNIPPETS = [
    "Excellent food and great ambiance.",
    "Best biryani in Colombo, highly recommend!",
    "Service could be faster but food is amazing.",
    "A hidden gem in the heart of the city.",
    "Fresh seafood, perfectly cooked.",
    "Cosy atmosphere, perfect for a date night.",
    "Very affordable and delicious.",
    "The crab curry is absolutely outstanding.",
    "Great variety in the buffet spread.",
    "Nice place but parking is difficult.",
    "Authentic flavours, reminded me of home cooking.",
    "The desserts are to die for!",
]


def fake_place_id(seed: str) -> str:
    return "ChIJ" + hashlib.md5(seed.encode()).hexdigest()[:20].upper()


def busyness_score(total_ratings: int, place_seed: str) -> float:
    hour = datetime.now().hour
    hour_curve = [20,15,12,10,10,12,18,30,45,60,65,70,80,85,75,70,75,85,90,88,80,65,50,35]
    hour_factor = hour_curve[hour] / 100.0
    log_pop = math.log1p(total_ratings) / math.log1p(5000)
    pop_factor = min(log_pop, 1.0)
    jitter_int = int(hashlib.md5(place_seed.encode()).hexdigest()[:4], 16)
    jitter = (jitter_int % 140 - 70) / 1000.0
    raw = (pop_factor * 0.6) + (hour_factor * 0.4) + jitter
    return round(min(max(raw, 0.0), 1.0) * 100, 2)


def generate_restaurants(n: int = 150) -> list:
    records = []
    for i in range(n):
        area_name, base_lat, base_lng = random.choice(AREAS)
        cuisine_type, _ = random.choice(CUISINES)
        prefix   = random.choice(NAME_PREFIXES)
        suffix   = random.choice(NAME_SUFFIXES)
        name     = f"{prefix} {suffix}"
        street   = random.choice(STREET_NAMES)
        number   = random.randint(1, 250)
        address  = f"{number} {street}, {area_name}, Sri Lanka"
        place_id = fake_place_id(f"{name}-{address}-{i}")

        lat = round(base_lat + random.uniform(-0.015, 0.015), 6)
        lng = round(base_lng + random.uniform(-0.015, 0.015), 6)

        rating        = round(random.uniform(3.0, 5.0), 1)
        total_ratings = random.randint(10, 4500)
        price_level   = random.randint(0, 4)
        open_now      = random.choice([True, False, None])

        # Pick 2–4 review snippets
        n_reviews   = random.randint(2, 4)
        review_texts = " | ".join(random.sample(REVIEW_SNIPPETS, n_reviews))
        avg_review   = round(random.uniform(3.0, 5.0), 1)

        phone   = f"+94 {random.choice(PHONE_PREFIXES)}-{random.randint(1000000,9999999)}"
        website = f"https://www.{name.lower().replace(' ', '')}.lk" if random.random() > 0.4 else ""

        types = f"{cuisine_type}, restaurant, food, establishment"

        busy = busyness_score(total_ratings, place_id)

        records.append({
            "place_id":           place_id,
            "name":               name,
            "address":            address,
            "lat":                lat,
            "lng":                lng,
            "rating":             rating,
            "total_ratings":      total_ratings,
            "price_level":        price_level,
            "busyness_score":     busy,
            "types":              types,
            "phone":              phone,
            "website":            website,
            "open_now":           open_now,
            "review_count":       n_reviews,
            "review_texts":       review_texts,
            "avg_review_rating":  avg_review,
            "collected_at":       datetime.now(timezone.utc).isoformat(),
        })

    return records


def main():
    records = generate_restaurants(150)
    fieldnames = list(records[0].keys())

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"✅  Mock data generated: {len(records)} restaurants → {OUTPUT_FILE}")

    # Write canonical raw CSV for 03_data_cleaning.py
    raw_path = ROOT / "data" / "raw" / "colombo_restaurants_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUTPUT_FILE, str(raw_path))
    print(f"✅  Raw data copy → {raw_path}")

    # Step 02: PySpark validation and Parquet output
    load_into_spark(OUTPUT_FILE, PARQUET_DIR)


if __name__ == "__main__":
    main()
