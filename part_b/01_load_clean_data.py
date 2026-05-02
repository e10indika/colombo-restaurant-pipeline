"""
part_b/01_load_clean_data.py
============================
Loads the cleaned restaurant CSV, validates all columns required by the
recommendation system, registers a Spark SQL temp view, runs confirmation
queries, and saves the validated data as the final dataset.

Run
---
    python part_b/01_load_clean_data.py
"""

import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common.spark_utils import build_spark
from common.paths import CLEAN_CSV, FINAL_CSV

# Columns the recommendation system depends on
REQUIRED_COLUMNS = [
    "place_id", "name", "cuisine_category", "district",
    "rating", "price_level", "popularity_score", "total_ratings",
]

# === SPARK SETUP ===
# build_spark imported from common.spark_utils


# === VALIDATION ===

def validate_columns(df) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for recommendation system: {missing}\n"
            f"Available columns: {df.columns}"
        )
    print(f"  ✓ All {len(REQUIRED_COLUMNS)} required columns present")


def print_summary(df) -> None:
    print(f"\n{'─'*50}")
    print("DATA SUMMARY")
    print(f"{'─'*50}")
    print(f"  Total restaurants : {df.count()}")
    print(f"  Columns           : {len(df.columns)}")
    print(f"  Schema:")
    for field in df.schema.fields:
        print(f"    {field.name:<25} {field.dataType.simpleString()}")
    print(f"\n  Null counts in required columns:")
    for col in REQUIRED_COLUMNS:
        n = df.filter(F.col(col).isNull()).count()
        status = "✓" if n == 0 else f"⚠  {n} nulls"
        print(f"    {col:<25} {status}")


# === SQL CONFIRMATION QUERIES ===

def run_confirmation_queries(spark) -> None:
    print(f"\n{'─'*50}")
    print("SPARK SQL CONFIRMATION QUERIES")
    print(f"{'─'*50}")

    print("\n  [Query 1] Count per cuisine_category:")
    spark.sql("""
        SELECT cuisine_category, COUNT(*) AS count
        FROM restaurants
        GROUP BY cuisine_category
        ORDER BY count DESC
    """).show(truncate=False)

    print("  [Query 2] Count per district:")
    spark.sql("""
        SELECT district, COUNT(*) AS count
        FROM restaurants
        GROUP BY district
        ORDER BY count DESC
    """).show(truncate=False)

    print("  [Query 3] Overall average rating:")
    spark.sql("""
        SELECT ROUND(AVG(rating), 4) AS overall_avg_rating,
               ROUND(AVG(popularity_score), 4) AS overall_avg_popularity
        FROM restaurants
    """).show(truncate=False)


# === MAIN ===

def main() -> None:
    print("=" * 60)
    print("PART B — Step 01: Load & Validate Clean Data")
    print("=" * 60)

    Path(FINAL_CSV).parent.mkdir(parents=True, exist_ok=True)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── Load ────────────────────────────────────────────────────────
    print(f"\n[Load] Reading {CLEAN_CSV}")
    try:
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(CLEAN_CSV)
        )
    except Exception as exc:
        print(f"[ERROR] Could not load clean CSV: {exc}", file=sys.stderr)
        spark.stop()
        sys.exit(1)

    # Explicit casts — inferSchema is unreliable on PySpark-written CSVs
    from pyspark.sql.types import DoubleType, IntegerType
    df = (
        df
        .withColumn("rating",            F.col("rating").cast(DoubleType()))
        .withColumn("total_ratings",     F.col("total_ratings").cast(IntegerType()))
        .withColumn("price_level",       F.col("price_level").cast(IntegerType()))
        .withColumn("popularity_score",  F.col("popularity_score").cast(DoubleType()))
        .withColumn("avg_review_rating", F.col("avg_review_rating").cast(DoubleType()))
    )

    df.cache()

    # ── Validate ────────────────────────────────────────────────────
    print("\n[Validate] Checking required columns …")
    try:
        validate_columns(df)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        spark.stop()
        sys.exit(1)

    print_summary(df)

    # ── Register Spark SQL temp view ────────────────────────────────
    df.createOrReplaceTempView("restaurants")
    print("\n[Spark SQL] Temp view 'restaurants' registered")

    run_confirmation_queries(spark)

    # ── Save final CSV ──────────────────────────────────────────────
    print(f"\n[Save] Writing final dataset → {FINAL_CSV}")
    (
        df
        .coalesce(1)
        .write
        .option("header", "true")
        .mode("overwrite")
        .csv(FINAL_CSV)
    )
    print(f"[Save] Done")

    spark.stop()
    print("\n✅  Final dataset ready for feature engineering.")


if __name__ == "__main__":
    main()
