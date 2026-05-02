"""
part_a/03_data_cleaning.py
==========================
Cleans raw Colombo restaurant data collected from Google Maps API.

Steps
-----
1. Load raw CSV with explicit schema
2. Deduplicate on place_id
3. Drop rows where both rating AND total_ratings are null
4. Impute nulls: median/mode/zero/empty-string strategies
5. Extract Colombo district from address
6. Normalise types column → cuisine_category
7. Add popularity_score and price_label derived columns
8. Save cleaned output to data/processed/colombo_restaurants_clean.csv

Run
---
    python part_a/03_data_cleaning.py
"""

import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType, DoubleType, IntegerType, StringType, StructField, StructType,
)

from common.spark_utils import build_spark
from common.paths import CLEAN_CSV, RAW_CSV
from common.constants import PRICE_LABELS


# === CONSTANTS ===

# Types to strip from the Google "types" list — too generic to be useful
_GENERIC_TYPES = {
    "point_of_interest", "establishment", "food", "store",
    "premise", "geocode", "locality", "political",
}

# Map the first meaningful Google type to a human-readable cuisine category
# (used in add_cuisine_category via F.when chain — no UDF needed)
_CUISINE_MAP = {
    "cafe":                "Cafe",
    "bakery":              "Bakery",
    "bar":                 "Bar",
    "night_club":          "Bar",
    "meal_takeaway":       "Takeaway",
    "meal_delivery":       "Takeaway",
    "fast_food_restaurant":"Fast Food",
    "pizza_restaurant":    "Pizza",
    "seafood_restaurant":  "Seafood",
    "chinese_restaurant":  "Chinese",
    "indian_restaurant":   "Indian",
    "italian_restaurant":  "Italian",
    "japanese_restaurant": "Japanese",
    "korean_restaurant":   "Korean",
    "thai_restaurant":     "Thai",
    "middle_eastern_restaurant": "Middle Eastern",
    "hamburger_restaurant":"Burgers",
    "sandwich_shop":       "Sandwiches",
    "ice_cream_shop":      "Desserts",
    "restaurant":          "Restaurant",
}

_PRICE_LABELS = PRICE_LABELS

# === SPARK SETUP ===
# build_spark imported from common.spark_utils


# === SCHEMA ===

def raw_schema() -> StructType:
    return StructType([
        StructField("place_id",          StringType(),  True),
        StructField("name",              StringType(),  True),
        StructField("address",           StringType(),  True),
        StructField("lat",               DoubleType(),  True),
        StructField("lng",               DoubleType(),  True),
        StructField("rating",            DoubleType(),  True),
        StructField("total_ratings",     IntegerType(), True),
        StructField("price_level",       IntegerType(), True),
        StructField("busyness_score",    DoubleType(),  True),   # pipeline-generated synthetic score
        StructField("types",             StringType(),  True),
        StructField("phone",             StringType(),  True),
        StructField("website",           StringType(),  True),
        StructField("open_now",          BooleanType(), True),
        StructField("review_count",      IntegerType(), True),
        StructField("review_texts",      StringType(),  True),
        StructField("avg_review_rating", DoubleType(),  True),
        StructField("collected_at",      StringType(),  True),   # ISO-8601 timestamp
    ])


# === NULL AUDIT ===

def print_null_counts(df, label: str) -> None:
    print(f"\n{'─'*50}")
    print(f"NULL COUNTS — {label}")
    print(f"{'─'*50}")
    null_row = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()
    for col, n in null_row.items():
        if n > 0:
            print(f"  {col:<25} {n:>5} nulls")
    print(f"  Total rows: {df.count()}")


# === IMPUTATION HELPERS ===

def _median(df, col: str) -> float:
    """Compute median of a numeric column (approx, 1% relative error)."""
    return df.approxQuantile(col, [0.5], 0.01)[0]


def _mode(df, col: str):
    """Return the most frequent non-null value of a column."""
    row = (
        df.filter(F.col(col).isNotNull())
          .groupBy(col).count()
          .orderBy(F.col("count").desc())
          .first()
    )
    return row[col] if row else None


# === DEDUPLICATION ===

def deduplicate(df):
    before = df.count()
    df = df.dropDuplicates(["place_id"])
    after = df.count()
    print(f"\n[Dedup] {before} → {after} rows  (removed {before - after} duplicates)")
    return df


# === DROP ROWS WHERE BOTH RATING AND TOTAL_RATINGS ARE NULL ===

def drop_unrateable(df):
    before = df.count()
    df = df.filter(~(F.col("rating").isNull() & F.col("total_ratings").isNull()))
    after = df.count()
    print(f"[Drop unrateable] {before} → {after} rows  (removed {before - after})")
    return df


# === NULL IMPUTATION ===

def impute_nulls(df):
    print("\n[Imputing nulls]")

    rating_median     = _median(df, "rating")
    review_median     = _median(df, "avg_review_rating")
    price_mode        = _mode(df, "price_level")

    print(f"  rating median          = {rating_median}")
    print(f"  avg_review_rating med  = {review_median}")
    print(f"  price_level mode       = {price_mode}")

    df = (
        df
        .fillna({"rating":            rating_median})
        .fillna({"total_ratings":     0})
        .fillna({"price_level":       int(price_mode) if price_mode is not None else 1})
        .fillna({"review_count":      0})
        .fillna({"avg_review_rating": review_median})
        .fillna({"review_texts":      ""})
    )
    return df


# === DISTRICT EXTRACTION ===

def extract_district(df):
    """
    Pull the Colombo district from the address field.
    Matches patterns like "Colombo 03", "Colombo 7", "Colombo 10".
    Falls back to "Colombo" when no district number is found.
    """
    district_pattern = r"(?i)Colombo\s*(\d{1,2})"
    df = df.withColumn(
        "district",
        F.when(
            F.regexp_extract(F.col("address"), district_pattern, 0) != "",
            F.regexp_replace(
                F.regexp_extract(F.col("address"), district_pattern, 0),
                r"(?i)Colombo\s*0?(\d+)",
                "Colombo $1",
            ),
        ).otherwise(F.lit("Colombo")),
    )
    print("\n[District extraction] district column added")
    return df


# === TYPE NORMALISATION + CUISINE CATEGORY ===
# Use pure Spark operations instead of UDFs to avoid pickle/serialisation issues.

def add_cuisine_category(df):
    """
    Derive primary_type and cuisine_category without Python UDFs.

    1. Split the comma-separated types string into an array.
    2. Remove generic/non-informative tags with array_except.
    3. Take the first remaining element as primary_type.
    4. Map primary_type → cuisine_category with a when/otherwise chain.
    """
    # Build a Spark array literal of all generic types to exclude
    generic_arr = F.array(*[F.lit(g) for g in sorted(_GENERIC_TYPES)])

    # Split "cafe, restaurant, food" → ["cafe", "restaurant", "food"]
    # and strip leading/trailing spaces from each element
    types_arr = F.transform(
        F.split(F.col("types"), r","),
        lambda t: F.trim(t),
    )

    # Drop all generic tags
    meaningful = F.array_except(types_arr, generic_arr)

    # First meaningful type, default "restaurant"
    primary = F.when(
        F.size(meaningful) > 0,
        F.element_at(meaningful, 1),
    ).otherwise(F.lit("restaurant"))

    df = df.withColumn("primary_type", primary)

    # Map primary_type → cuisine_category
    cuisine = (
        F.when(F.col("primary_type") == "cafe",                    "Cafe")
         .when(F.col("primary_type") == "bakery",                  "Bakery")
         .when(F.col("primary_type") == "bar",                     "Bar")
         .when(F.col("primary_type") == "night_club",              "Bar")
         .when(F.col("primary_type") == "meal_takeaway",           "Takeaway")
         .when(F.col("primary_type") == "meal_delivery",           "Takeaway")
         .when(F.col("primary_type") == "fast_food_restaurant",    "Fast Food")
         .when(F.col("primary_type") == "pizza_restaurant",        "Pizza")
         .when(F.col("primary_type") == "seafood_restaurant",      "Seafood")
         .when(F.col("primary_type") == "chinese_restaurant",      "Chinese")
         .when(F.col("primary_type") == "indian_restaurant",       "Indian")
         .when(F.col("primary_type") == "italian_restaurant",      "Italian")
         .when(F.col("primary_type") == "japanese_restaurant",     "Japanese")
         .when(F.col("primary_type") == "korean_restaurant",       "Korean")
         .when(F.col("primary_type") == "thai_restaurant",         "Thai")
         .when(F.col("primary_type") == "middle_eastern_restaurant","Middle Eastern")
         .when(F.col("primary_type") == "hamburger_restaurant",    "Burgers")
         .when(F.col("primary_type") == "sandwich_shop",           "Sandwiches")
         .when(F.col("primary_type") == "ice_cream_shop",          "Desserts")
         .when(F.col("primary_type") == "restaurant",              "Restaurant")
         .otherwise("Other")
    )

    df = df.withColumn("cuisine_category", cuisine)
    print("[Cuisine category] primary_type and cuisine_category columns added")
    return df


# === DERIVED COLUMNS ===

def add_popularity_score(df):
    """popularity_score = log1p(total_ratings) × rating"""
    df = df.withColumn(
        "popularity_score",
        F.round(F.log1p(F.col("total_ratings").cast(DoubleType())) * F.col("rating"), 4),
    )
    print("[Popularity score] popularity_score column added")
    return df


def add_price_label(df):
    """Map integer price_level to human-readable label."""
    df = df.withColumn(
        "price_label",
        F.when(F.col("price_level") == 0, "Free")
         .when(F.col("price_level") == 1, "Budget")
         .when(F.col("price_level") == 2, "Moderate")
         .when(F.col("price_level") == 3, "Expensive")
         .when(F.col("price_level") == 4, "Luxury")
         .otherwise("Unknown"),
    )
    print("[Price label] price_label column added")
    return df


# === FINAL TYPE CAST ===

def cast_types(df):
    df = (
        df
        .withColumn("lat",               F.col("lat").cast(DoubleType()))
        .withColumn("lng",               F.col("lng").cast(DoubleType()))
        .withColumn("rating",            F.col("rating").cast(DoubleType()))
        .withColumn("total_ratings",     F.col("total_ratings").cast(IntegerType()))
        .withColumn("price_level",       F.col("price_level").cast(IntegerType()))
        .withColumn("review_count",      F.col("review_count").cast(IntegerType()))
        .withColumn("avg_review_rating", F.col("avg_review_rating").cast(DoubleType()))
        .withColumn("popularity_score",  F.col("popularity_score").cast(DoubleType()))
    )
    return df


# === SAVE ===

def save_clean(df, path: str) -> None:
    print(f"\n[Save] Writing cleaned data → {path}")
    (
        df
        .coalesce(1)
        .write
        .option("header", "true")
        .mode("overwrite")
        .csv(path)
    )
    print(f"[Save] Done — {df.count()} rows written")


# === MAIN ===

def main() -> None:
    print("=" * 60)
    print("PART A — Step 03: Data Cleaning")
    print("=" * 60)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── Load raw data ──────────────────────────────────────────────
    print(f"\n[Load] Reading {RAW_CSV}")
    try:
        df = (
            spark.read
            .option("header", "true")
            .option("nullValue", "")
            .option("multiLine", "true")
            .option("escape", '"')
            .schema(raw_schema())
            .csv(RAW_CSV)
        )
    except Exception as exc:
        print(f"[ERROR] Could not load raw CSV: {exc}", file=sys.stderr)
        spark.stop()
        sys.exit(1)

    df.cache()
    print_null_counts(df, "BEFORE CLEANING")

    # ── Cleaning pipeline ─────────────────────────────────────────
    df = deduplicate(df)
    df = drop_unrateable(df)
    df = impute_nulls(df)
    df = extract_district(df)
    df = add_cuisine_category(df)
    df = add_popularity_score(df)
    df = add_price_label(df)
    df = cast_types(df)

    df.cache()
    print_null_counts(df, "AFTER CLEANING")

    # ── Preview ───────────────────────────────────────────────────
    print("\n[Preview] Sample rows:")
    df.select(
        "name", "district", "cuisine_category",
        "rating", "total_ratings", "popularity_score", "price_label",
    ).show(10, truncate=False)

    # ── Save ──────────────────────────────────────────────────────
    save_clean(df, CLEAN_CSV)

    spark.stop()
    print("\n✅  Data cleaning complete.")


if __name__ == "__main__":
    main()
