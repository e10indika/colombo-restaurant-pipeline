"""
part_b/02_feature_engineering.py
==================================
Engineers features from the final restaurant dataset for the ALS
recommendation model.

Features created
----------------
- cuisine_index       : StringIndexer on cuisine_category
- district_index      : StringIndexer on district
- rating_normalized   : MinMaxScaler(rating)
- popularity_normalized: MinMaxScaler(popularity_score)
- features            : VectorAssembler of all numeric features
- user_id             : synthetic 1–200 user IDs
- interaction_score   : 1–5 integer rating simulating user behaviour

Outputs
-------
- data/final/colombo_restaurants_features.csv
- models/feature_pipeline/   (fitted Pipeline model)

Run
---
    python part_b/02_feature_engineering.py
"""

import sys
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from common.spark_utils import build_spark
from common.paths import FINAL_CSV, FEATURES_CSV, FEATURE_PIPELINE

# === SPARK SETUP ===
# build_spark imported from common.spark_utils


# === FEATURE ENGINEERING STEPS ===

def add_synthetic_user_id(df):
    """
    Assign random user IDs 1–200 to simulate 200 distinct users.
    Uses monotonically_increasing_id as a stable per-row seed.
    """
    df = df.withColumn(
        "user_id",
        (F.monotonically_increasing_id() % 200 + 1).cast(IntegerType()),
    )
    print("  [user_id] Synthetic user IDs 1–200 added")
    return df


def build_pipeline(df):
    """
    Returns a fitted Pipeline containing:
      1. StringIndexer  → cuisine_index
      2. StringIndexer  → district_index
      3. VectorAssembler for rating (single feature) → MinMaxScaler
      4. VectorAssembler for popularity  → MinMaxScaler
      5. Final VectorAssembler → features vector
    """

    # ── Stage 1 & 2: Categorical encoding ───────────────────────────
    cuisine_indexer = StringIndexer(
        inputCol="cuisine_category",
        outputCol="cuisine_index",
        handleInvalid="keep",
    )
    district_indexer = StringIndexer(
        inputCol="district",
        outputCol="district_index",
        handleInvalid="keep",
    )

    # ── Stage 3: MinMaxScaler for rating ────────────────────────────
    rating_assembler = VectorAssembler(
        inputCols=["rating"],
        outputCol="_rating_vec",
        handleInvalid="keep",
    )
    rating_scaler = MinMaxScaler(
        inputCol="_rating_vec",
        outputCol="rating_normalized",
    )

    # ── Stage 4: MinMaxScaler for popularity_score ─────────────────
    pop_assembler = VectorAssembler(
        inputCols=["popularity_score"],
        outputCol="_pop_vec",
        handleInvalid="keep",
    )
    pop_scaler = MinMaxScaler(
        inputCol="_pop_vec",
        outputCol="popularity_normalized",
    )

    # ── Stage 5: Final feature vector ───────────────────────────────
    feature_assembler = VectorAssembler(
        inputCols=[
            "rating_normalized",
            "popularity_normalized",
            "cuisine_index",
            "district_index",
            "price_level",
        ],
        outputCol="features",
        handleInvalid="keep",
    )

    pipeline = Pipeline(stages=[
        cuisine_indexer,
        district_indexer,
        rating_assembler,
        rating_scaler,
        pop_assembler,
        pop_scaler,
        feature_assembler,
    ])

    print("  [Pipeline] Fitting feature pipeline …")
    model = pipeline.fit(df)
    print("  [Pipeline] Fit complete")
    return model


def add_interaction_score(df):
    """
    Simulate a 1–5 integer rating from rating_normalized.
    Uses vector_to_array to safely extract the scalar from the DenseVector,
    then scales to [1, 5] and rounds to an integer.
    """
    df = df.withColumn(
        "interaction_score",
        F.greatest(
            F.lit(1),
            F.least(
                F.lit(5),
                F.round(
                    vector_to_array(F.col("rating_normalized"))[0] * 4 + 1,
                    0,
                ).cast(IntegerType()),
            ),
        ),
    )
    print("  [interaction_score] Synthetic 1–5 interaction scores added")
    return df


def save_features(df, path: str) -> None:
    """Save only the scalar columns needed by the ALS model and recommendation layer.
    Vector-type columns (rating_normalized, popularity_normalized, features)
    are excluded — they serialise as unreadable strings in CSV.
    """
    cols_to_save = [
        "place_id", "name", "cuisine_category", "district",
        "rating", "total_ratings", "price_level", "price_label",
        "popularity_score", "cuisine_index", "district_index",
        "user_id", "interaction_score",
    ]
    existing = [c for c in cols_to_save if c in df.columns]
    print(f"\n[Save] Writing features CSV → {path}")
    (
        df.select(*existing)
          .coalesce(1)
          .write
          .option("header", "true")
          .mode("overwrite")
          .csv(path)
    )
    print(f"[Save] Done")


# === MAIN ===

def main() -> None:
    print("=" * 60)
    print("PART B — Step 02: Feature Engineering")
    print("=" * 60)

    Path(FEATURE_PIPELINE).parent.mkdir(parents=True, exist_ok=True)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── Load final dataset ──────────────────────────────────────────
    print(f"\n[Load] Reading {FINAL_CSV}")
    try:
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(FINAL_CSV)
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        spark.stop()
        sys.exit(1)

    df.cache()
    print(f"  Rows: {df.count()}")

    # Explicit casts — inferSchema is unreliable on PySpark-written CSVs
    df = (
        df
        .withColumn("rating",           F.col("rating").cast("double"))
        .withColumn("total_ratings",    F.col("total_ratings").cast("integer"))
        .withColumn("popularity_score", F.col("popularity_score").cast("double"))
        .withColumn("price_level",      F.col("price_level").cast("double").cast("integer"))
    )

    # Fill any remaining nulls in columns used by VectorAssembler
    rating_median     = df.approxQuantile("rating", [0.5], 0.01)[0] or 4.0
    pop_median        = df.approxQuantile("popularity_score", [0.5], 0.01)[0] or 0.0
    df = df.fillna({
        "rating":           rating_median,
        "popularity_score": pop_median,
        "price_level":      1,
    })

    # ── Synthetic users ─────────────────────────────────────────────
    print("\n[Step 1] Adding synthetic user IDs")
    df = add_synthetic_user_id(df)

    # ── Feature pipeline ────────────────────────────────────────────
    print("\n[Step 2] Building and fitting ML Pipeline")
    pipeline_model = build_pipeline(df)
    df_transformed = pipeline_model.transform(df)

    # ── Interaction score ───────────────────────────────────────────
    print("\n[Step 3] Adding interaction_score")
    df_transformed = add_interaction_score(df_transformed)

    # ── Preview ─────────────────────────────────────────────────────
    print("\n[Preview] Sample feature rows:")
    df_transformed.select(
        "name", "user_id", "cuisine_index", "district_index",
        "rating_normalized", "interaction_score",
    ).show(5, truncate=False)

    # ── Save features CSV ───────────────────────────────────────────
    save_features(df_transformed, FEATURES_CSV)

    # ── Save fitted Pipeline model ──────────────────────────────────
    print(f"\n[Save] Pipeline model → {FEATURE_PIPELINE}")
    pipeline_model.write().overwrite().save(FEATURE_PIPELINE)
    print(f"[Save] Done")

    spark.stop()
    print("\n✅  Feature engineering complete.")


if __name__ == "__main__":
    main()
