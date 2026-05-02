"""
part_b/04_recommendations.py
==============================
Loads the trained ALS model and generates personalised restaurant
recommendations for all users, then provides a helper function to
retrieve and display recommendations for a specific user.

Outputs
-------
- data/final/all_user_recommendations.csv

Run
---
    python part_b/04_recommendations.py
"""

import sys
from pathlib import Path

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common.spark_utils import build_spark
from common.restaurant_index import add_restaurant_index
from common.paths import CLEAN_CSV, FEATURES_CSV, ALS_MODEL_DIR, RECS_CSV

TOP_N = 10   # recommendations per user


# === SPARK SETUP ===
# build_spark imported from common.spark_utils


# === RESTAURANT INDEX (must match training) ===
# add_restaurant_index imported from common.restaurant_index


# === METADATA LOOKUP ===

def build_metadata(df_features, df_clean):
    """
    Join features data with clean data to produce a metadata lookup:
    restaurant_index → name, cuisine_category, district, rating, price_label.
    """
    meta_cols = ["place_id", "name", "cuisine_category", "district",
                 "rating", "price_label"]
    existing  = [c for c in meta_cols if c in df_clean.columns]

    metadata = (
        df_features.select("place_id", "restaurant_index")
                   .join(df_clean.select(*existing), on="place_id", how="left")
                   .dropDuplicates(["restaurant_index"])
    )
    return metadata


# === GENERATE RECOMMENDATIONS ===

def generate_all_recommendations(model, metadata):
    """
    Use ALS recommendForAllUsers() to get top-N recs per user,
    explode the array, then join metadata for readable output.
    """
    print(f"\n[Step] Generating top {TOP_N} recommendations for all users …")
    raw_recs = model.recommendForAllUsers(TOP_N)

    # Explode nested recommendations array
    recs = (
        raw_recs
        .select(
            F.col("user_id"),
            F.explode(F.col("recommendations")).alias("rec"),
        )
        .select(
            F.col("user_id"),
            F.col("rec.restaurant_index").alias("restaurant_index"),
            F.round(F.col("rec.rating"), 4).alias("predicted_score"),
        )
    )

    # Join metadata
    recs_with_meta = (
        recs.join(metadata, on="restaurant_index", how="left")
            .select(
                "user_id", "restaurant_index", "predicted_score",
                "name", "cuisine_category", "district",
                "rating", "price_label",
            )
            .orderBy("user_id", F.col("predicted_score").desc())
    )

    print(f"  Total recommendation rows: {recs_with_meta.count()}")
    return recs_with_meta


# === USER-LEVEL HELPER ===

def get_recommendations(recs_df, user_id: int) -> None:
    """
    Print the top-N recommendations for a specific user in a readable table.

    Parameters
    ----------
    recs_df : Spark DataFrame  — full recommendations DataFrame
    user_id : int              — target user ID (1–200)
    """
    user_recs = (
        recs_df
        .filter(F.col("user_id") == user_id)
        .orderBy(F.col("predicted_score").desc())
        .select("name", "cuisine_category", "district", "rating",
                "price_label", "predicted_score")
        .collect()
    )

    print(f"\n{'═'*70}")
    print(f"  🍽️  Top {TOP_N} Recommendations for User {user_id}")
    print(f"{'═'*70}")

    if not user_recs:
        print("  No recommendations found for this user (cold start).")
        return

    print(f"  {'#':<4} {'Restaurant':<30} {'Cuisine':<14} {'District':<12} {'Rating':<7} {'Price':<10} {'Score'}")
    print(f"  {'─'*4} {'─'*30} {'─'*14} {'─'*12} {'─'*7} {'─'*10} {'─'*6}")
    for i, row in enumerate(user_recs, 1):
        print(
            f"  {i:<4} {str(row['name']):<30} {str(row['cuisine_category']):<14} "
            f"{str(row['district']):<12} {str(row['rating']):<7} "
            f"{str(row['price_label']):<10} {row['predicted_score']:.4f}"
        )


# === MAIN ===

def main() -> None:
    print("=" * 60)
    print("PART B — Step 04: Recommendations")
    print("=" * 60)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── Load features ───────────────────────────────────────────────
    print(f"\n[Load] Features → {FEATURES_CSV}")
    try:
        df_features = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(FEATURES_CSV)
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        spark.stop()
        sys.exit(1)

    df_features = df_features.withColumn(
        "user_id", F.col("user_id").cast("integer")
    )

    # ── Load clean data (for metadata) ─────────────────────────────
    print(f"[Load] Clean data → {CLEAN_CSV}")
    try:
        df_clean = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(CLEAN_CSV)
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        spark.stop()
        sys.exit(1)

    # ── Rebuild restaurant index (same as training) ─────────────────
    df_features, _ = add_restaurant_index(df_features)
    df_features.cache()

    # ── Build metadata lookup ───────────────────────────────────────
    metadata = build_metadata(df_features, df_clean)
    metadata.cache()

    # ── Load ALS model ──────────────────────────────────────────────
    print(f"\n[Load] ALS model → {ALS_MODEL_DIR}")
    try:
        als_model = ALSModel.load(ALS_MODEL_DIR)
    except Exception as exc:
        print(f"[ERROR] Could not load ALS model: {exc}", file=sys.stderr)
        spark.stop()
        sys.exit(1)

    # ── Generate all recommendations ────────────────────────────────
    all_recs = generate_all_recommendations(als_model, metadata)
    all_recs.cache()

    # ── Demo: specific users ────────────────────────────────────────
    for uid in [1, 5, 10]:
        get_recommendations(all_recs, uid)

    # ── Save ────────────────────────────────────────────────────────
    print(f"\n[Save] All recommendations → {RECS_CSV}")
    (
        all_recs
        .coalesce(1)
        .write
        .option("header", "true")
        .mode("overwrite")
        .csv(RECS_CSV)
    )
    print("[Save] Done")

    spark.stop()
    print("\n✅  Recommendation generation complete.")


if __name__ == "__main__":
    main()
