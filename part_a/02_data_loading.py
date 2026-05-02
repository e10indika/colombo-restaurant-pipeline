"""
part_a/02_data_loading.py
==========================
Step 02 — PySpark Data Loading, Validation, and Parquet Persistence.

Loads the raw restaurant CSV produced by 01_data_collection.py into
PySpark, validates the schema, audits nulls, surfaces quick insights,
and saves a Parquet snapshot for fast downstream reads.

Can be imported as a module (load_into_spark) or run standalone:

    python part_a/02_data_loading.py [csv_path] [parquet_dir]

Public API
----------
    load_into_spark(csv_path: str, parquet_dir: str) -> None
"""

import logging
import os

from common.spark_utils import build_spark

logger = logging.getLogger(__name__)


# ── Schema definition ──────────────────────────────────────────────────────────

def _restaurant_schema():
    from pyspark.sql.types import (
        FloatType, IntegerType, StringType, StructField, StructType,
    )
    return StructType([
        StructField("place_id",          StringType(),  True),
        StructField("name",              StringType(),  True),
        StructField("address",           StringType(),  True),
        StructField("lat",               FloatType(),   True),
        StructField("lng",               FloatType(),   True),
        StructField("rating",            FloatType(),   True),
        StructField("total_ratings",     IntegerType(), True),
        StructField("price_level",       IntegerType(), True),
        StructField("busyness_score",    FloatType(),   True),
        StructField("types",             StringType(),  True),
        StructField("phone",             StringType(),  True),
        StructField("website",           StringType(),  True),
        StructField("open_now",          StringType(),  True),  # "True"/"False"/None
        StructField("review_count",      IntegerType(), True),
        StructField("review_texts",      StringType(),  True),
        StructField("avg_review_rating", FloatType(),   True),
        StructField("collected_at",      StringType(),  True),
    ])


# ── SparkSession factory ────────────────────────────────────────────────────────
# build_spark imported from common.spark_utils


# ── Validation helpers ─────────────────────────────────────────────────────────

def _log_schema(df) -> None:
    logger.info("── Schema")
    for field in df.schema.fields:
        logger.info("  %-22s %s", field.name, field.dataType.simpleString())


def _null_audit(df, total_rows: int) -> None:
    from pyspark.sql import functions as F

    logger.info("── Null audit  (column → null count)")
    null_counts = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()

    for col_name, n in null_counts.items():
        if n > 0:
            pct = round(n / total_rows * 100, 1)
            logger.warning("  %-22s %4d nulls  (%s%%)", col_name, n, pct)
        else:
            logger.info("  %-22s  0 nulls", col_name)


def _descriptive_stats(df) -> None:
    numeric_cols = ["rating", "total_ratings", "busyness_score", "price_level"]
    logger.info("── Descriptive statistics")
    stats = df.select(*numeric_cols).describe().collect()
    header = ["stat"] + numeric_cols
    logger.info("  %s", "  ".join(f"{h:<18}" for h in header))
    for row in stats:
        vals = [row["summary"]] + [
            str(row[c])[:16] if row[c] is not None else "N/A"
            for c in numeric_cols
        ]
        logger.info("  %s", "  ".join(f"{v:<18}" for v in vals))


# ── Insights ───────────────────────────────────────────────────────────────────

def _log_insights(df) -> None:
    from pyspark.sql import functions as F

    logger.info("── Quick insights")

    top_rated = (
        df.filter(F.col("total_ratings") >= 50)
          .orderBy(F.col("rating").desc())
          .select("name", "rating", "total_ratings", "address")
          .limit(5)
          .collect()
    )
    logger.info("  Top 5 highest-rated (min 50 reviews):")
    for r in top_rated:
        logger.info("    %-35s  ★ %-4s  (%d reviews)", r["name"], r["rating"], r["total_ratings"])

    busiest = (
        df.orderBy(F.col("busyness_score").desc())
          .select("name", "busyness_score", "address")
          .limit(5)
          .collect()
    )
    logger.info("  Top 5 busiest right now:")
    for r in busiest:
        logger.info("    %-35s  %s%%", r["name"], r["busyness_score"])

    _PRICE_LABELS = {
        0: "Budget", 1: "Inexpensive", 2: "Moderate",
        3: "Expensive", 4: "Very Expensive",
    }
    price_dist = (
        df.groupBy("price_level")
          .count()
          .orderBy("price_level")
          .collect()
    )
    logger.info("  Price level distribution:")
    for r in price_dist:
        label = _PRICE_LABELS.get(r["price_level"], "Unknown")
        logger.info("    Level %-2s  %-14s  %d restaurants",
                    r["price_level"], f"({label})", r["count"])


# ── Parquet writer ─────────────────────────────────────────────────────────────

def _save_parquet(df, parquet_dir: str) -> None:
    os.makedirs(os.path.dirname(parquet_dir) or ".", exist_ok=True)
    (
        df
        .repartition(1)       # single output file — dataset is small
        .write
        .mode("overwrite")
        .parquet(parquet_dir)
    )
    logger.info("Parquet saved → %s", parquet_dir)


# ── Public entry point ─────────────────────────────────────────────────────────

def load_into_spark(csv_path: str, parquet_dir: str) -> None:
    """
    Load a restaurant CSV into PySpark, validate it, surface insights,
    and persist the result as Parquet.

    Parameters
    ----------
    csv_path    Path to the snapshot CSV (just written by the pipeline).
    parquet_dir Output directory for the Parquet dataset.
    """
    logger.info("── Spark loading step ──────────────────────────────────")

    spark = build_spark("ColomboRestaurantLoader", "1g")
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession started  (local[*])")
    try:
        df = (
            spark.read
            .option("header", "true")
            .option("nullValue", "")
            .schema(_restaurant_schema())
            .csv(csv_path)
        )
        df.cache()

        total_rows = df.count()
        logger.info("Rows loaded: %d", total_rows)

        _log_schema(df)
        _null_audit(df, total_rows)
        _descriptive_stats(df)
        _log_insights(df)
        _save_parquet(df, parquet_dir)

        df.unpersist()
    finally:
        spark.stop()

    logger.info("── Spark loading step complete ─────────────────────────")


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).resolve().parent.parent
    _csv  = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "colombo_restaurants.csv")
    _parq = sys.argv[2] if len(sys.argv) > 2 else str(_ROOT / "data" / "restaurants.parquet")

    print(f"[02_data_loading] CSV  → {_csv}")
    print(f"[02_data_loading] Parq → {_parq}")
    load_into_spark(_csv, _parq)
