"""
part_b/05_evaluation.py
========================
Evaluates the trained ALS recommendation model using:
- RMSE  (Root Mean Square Error)
- MAE   (Mean Absolute Error)
- Coverage  — % of restaurants that appear in ≥1 recommendation
- Diversity — average number of unique cuisine categories per user's top-10

Outputs
-------
- data/final/evaluation_metrics.json

Run
---
    python part_b/05_evaluation.py
"""

import json
import sys
from pathlib import Path

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common.spark_utils import build_spark
from common.restaurant_index import add_restaurant_index
from common.paths import (
    FEATURES_CSV, ALS_MODEL_DIR, METRICS_JSON, CLEAN_CSV,
    ALS_RANK, ALS_ITER, ALS_REG, TRAIN_SEED, TRAIN_SPLIT,
)

# === SPARK SETUP ===
# build_spark imported from common.spark_utils


# === RESTAURANT INDEX ===
# add_restaurant_index imported from common.restaurant_index


# === RATING METRICS ===

def evaluate_rating_metrics(model, test):
    """Compute RMSE and MAE on the test split."""
    predictions = model.transform(test).filter(F.col("prediction").isNotNull())

    evaluator_rmse = RegressionEvaluator(
        metricName="rmse",
        labelCol="interaction_score",
        predictionCol="prediction",
    )
    evaluator_mae = RegressionEvaluator(
        metricName="mae",
        labelCol="interaction_score",
        predictionCol="prediction",
    )

    rmse = evaluator_rmse.evaluate(predictions)
    mae  = evaluator_mae.evaluate(predictions)

    print(f"\n{'─'*50}")
    print("RATING METRICS")
    print(f"{'─'*50}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")

    return rmse, mae


# === COVERAGE ===

def compute_coverage(model, df):
    """
    Coverage = (unique restaurants in recommendations) /
               (total restaurants in dataset) × 100
    """
    total_restaurants = df.select("restaurant_index").distinct().count()

    recs = model.recommendForAllUsers(10)
    rec_restaurant_ids = (
        recs.select(F.explode("recommendations").alias("rec"))
            .select(F.col("rec.restaurant_index").alias("restaurant_index"))
            .distinct()
    )
    covered = rec_restaurant_ids.count()
    coverage_pct = round(covered / total_restaurants * 100, 2)

    print(f"\n{'─'*50}")
    print("COVERAGE")
    print(f"{'─'*50}")
    print(f"  Total restaurants    : {total_restaurants}")
    print(f"  Restaurants covered  : {covered}")
    print(f"  Coverage             : {coverage_pct}%")

    return coverage_pct, recs


# === DIVERSITY ===

def compute_diversity(recs, df_features, df_clean):
    """
    Diversity = average number of unique cuisine_categories in each
    user's top-10 recommendations.

    Higher → model recommends across a wider variety of cuisines.
    """
    # Rebuild metadata: restaurant_index → cuisine_category
    cuisine_lookup = (
        df_features.select("place_id", "restaurant_index")
                   .join(
                       df_clean.select("place_id", "cuisine_category"),
                       on="place_id", how="left",
                   )
                   .dropDuplicates(["restaurant_index"])
                   .select("restaurant_index", "cuisine_category")
    )

    # Explode recs, join cuisine
    recs_exploded = (
        recs.select(
            F.col("user_id"),
            F.explode("recommendations").alias("rec"),
        )
        .select(
            "user_id",
            F.col("rec.restaurant_index").alias("restaurant_index"),
        )
        .join(cuisine_lookup, on="restaurant_index", how="left")
    )

    # Per-user distinct cuisine count
    diversity_per_user = (
        recs_exploded
        .groupBy("user_id")
        .agg(F.countDistinct("cuisine_category").alias("unique_cuisines"))
    )

    avg_diversity = diversity_per_user.agg(
        F.round(F.avg("unique_cuisines"), 4).alias("avg_diversity")
    ).collect()[0]["avg_diversity"]

    print(f"\n{'─'*50}")
    print("DIVERSITY")
    print(f"{'─'*50}")
    print(f"  Avg unique cuisines per user's top-10 : {avg_diversity}")

    return float(avg_diversity)


# === SAVE METRICS ===

def save_metrics(metrics: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Save] Metrics → {path}")


# === MAIN ===

def main() -> None:
    print("=" * 60)
    print("PART B — Step 05: Model Evaluation")
    print("=" * 60)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── Load features ───────────────────────────────────────────────
    print(f"\n[Load] Features → {FEATURES_CSV}")
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

    df = (
        df
        .withColumn("user_id",          F.col("user_id").cast("integer"))
        .withColumn("interaction_score", F.col("interaction_score").cast("float"))
    )

    df_clean = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(CLEAN_CSV)
    )

    # ── Recreate same index as training ─────────────────────────────
    df, _ = add_restaurant_index(df)
    df.cache()

    # ── Same split as training (seed=42) ────────────────────────────
    _, test = df.randomSplit(TRAIN_SPLIT, seed=TRAIN_SEED)
    test.cache()
    print(f"\n  Test rows: {test.count()}")

    # ── Load model ──────────────────────────────────────────────────
    print(f"\n[Load] ALS model → {ALS_MODEL_DIR}")
    try:
        als_model = ALSModel.load(ALS_MODEL_DIR)
    except Exception as exc:
        print(f"[ERROR] Could not load ALS model: {exc}", file=sys.stderr)
        spark.stop()
        sys.exit(1)

    # ── Evaluate ────────────────────────────────────────────────────
    rmse, mae            = evaluate_rating_metrics(als_model, test)
    coverage_pct, recs   = compute_coverage(als_model, df)
    avg_diversity        = compute_diversity(recs, df, df_clean)

    # ── Summary ─────────────────────────────────────────────────────
    metrics = {
        "rmse":              round(rmse, 6),
        "mae":               round(mae, 6),
        "coverage_percent":  coverage_pct,
        "avg_diversity":     avg_diversity,
        "als_params": {
            "rank":    ALS_RANK,
            "maxIter": ALS_ITER,
            "regParam": ALS_REG,
        },
    }

    print(f"\n{'═'*50}")
    print("EVALUATION SUMMARY")
    print(f"{'═'*50}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")

    save_metrics(metrics, METRICS_JSON)

    spark.stop()
    print("\n✅  Evaluation complete.")


if __name__ == "__main__":
    main()
