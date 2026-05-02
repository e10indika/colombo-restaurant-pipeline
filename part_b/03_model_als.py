"""
part_b/03_model_als.py
=======================
Trains an ALS (Alternating Least Squares) collaborative filtering model
on the feature-engineered Colombo restaurant dataset.

ALS treats each restaurant as an "item" and each synthetic user as a
"user".  The interaction_score (1–5) simulates explicit feedback.

Outputs
-------
- models/als_model/   (trained ALS model saved to disk)

Run
---
    python part_b/03_model_als.py
"""

import sys
from pathlib import Path

from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common.spark_utils import build_spark
from common.restaurant_index import add_restaurant_index
from common.paths import FEATURES_CSV, ALS_MODEL_DIR, ALS_RANK, ALS_ITER, ALS_REG, TRAIN_SEED, TRAIN_SPLIT

# === ALS HYPER-PARAMETERS ===
ALS_PARAMS = dict(
    rank             = ALS_RANK,
    maxIter          = ALS_ITER,
    regParam         = ALS_REG,
    userCol          = "user_id",
    itemCol          = "restaurant_index",
    ratingCol        = "interaction_score",
    coldStartStrategy= "drop",
    implicitPrefs    = False,
    seed             = TRAIN_SEED,
)


# === SPARK SETUP ===
# build_spark imported from common.spark_utils


# === RESTAURANT INDEX ===
# add_restaurant_index imported from common.restaurant_index


# === TRAIN / TEST SPLIT ===

def split_data(df):
    print(f"\n[Step 2] Splitting data 80% train / 20% test (seed={TRAIN_SEED})")
    train, test = df.randomSplit(TRAIN_SPLIT, seed=TRAIN_SEED)
    train.cache()
    test.cache()
    print(f"  Train rows: {train.count()}")
    print(f"  Test rows : {test.count()}")
    return train, test


# === ALS TRAINING ===

def train_als(train):
    print("\n[Step 3] Training ALS model …")
    print(f"  Parameters: {ALS_PARAMS}")
    als   = ALS(**ALS_PARAMS)
    model = als.fit(train)
    print("  ✓ ALS training complete")
    return model


# === PREDICTION SAMPLE ===

def show_predictions(model, test) -> None:
    print("\n[Step 4] Sample predictions on test set:")
    preds = model.transform(test)
    preds.select(
        "user_id", "restaurant_index", "interaction_score", "prediction"
    ).show(10, truncate=False)


# === MAIN ===

def main() -> None:
    print("=" * 60)
    print("PART B — Step 03: ALS Model Training")
    print("=" * 60)

    Path(ALS_MODEL_DIR).parent.mkdir(parents=True, exist_ok=True)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── Load features ───────────────────────────────────────────────
    print(f"\n[Load] Reading {FEATURES_CSV}")
    try:
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(FEATURES_CSV)
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        spark.stop()
        sys.exit(1)

    df.cache()
    print(f"  Rows: {df.count()}")

    # ── Ensure correct types ────────────────────────────────────────
    df = (
        df
        .withColumn("user_id",          F.col("user_id").cast("integer"))
        .withColumn("interaction_score", F.col("interaction_score").cast("float"))
    )

    # ── Restaurant index ────────────────────────────────────────────
    df, _indexer_model = add_restaurant_index(df)
    print(f"  Unique restaurants indexed: {df.select('restaurant_index').distinct().count()}")

    # ── Split ────────────────────────────────────────────────────────
    train, test = split_data(df)

    # ── Train ────────────────────────────────────────────────────────
    als_model = train_als(train)

    # ── Sample predictions ───────────────────────────────────────────
    show_predictions(als_model, test)

    # ── Save model ───────────────────────────────────────────────────
    print(f"\n[Save] ALS model → {ALS_MODEL_DIR}")
    als_model.write().overwrite().save(ALS_MODEL_DIR)
    print(f"[Save] Done")

    spark.stop()
    print("\n✅  ALS model training complete.")


if __name__ == "__main__":
    main()
