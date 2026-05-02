"""
part_a/04_eda_analytics.py
==========================
Exploratory Data Analysis on the cleaned Colombo restaurant dataset.

Performs 15 named analyses, prints results, and saves each result
as its own CSV under data/processed/analytics/.

Run
---
    python part_a/04_eda_analytics.py
"""

import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common.spark_utils import build_spark
from common.paths import CLEAN_CSV, ANALYTICS_DIR

ANALYTICS_DIR = Path(ANALYTICS_DIR)


# === SPARK SETUP ===
# build_spark imported from common.spark_utils


# === HELPERS ===

def section(num: int, title: str) -> None:
    print(f"\n{'='*50}")
    print(f"ANALYSIS {num}: {title}")
    print(f"{'='*50}")


def save_csv(df, name: str) -> None:
    out = str(ANALYTICS_DIR / name)
    df.coalesce(1).write.option("header", "true").mode("overwrite").csv(out)
    print(f"  [Saved] {out}")


# === ANALYSES ===

def analysis_01_cuisine_count(df):
    section(1, "Restaurant Count by Cuisine Category")
    result = df.groupBy("cuisine_category").count().orderBy(F.col("count").desc())
    result.show(truncate=False)
    save_csv(result, "01_cuisine_count")
    return result


def analysis_02_district_count(df):
    section(2, "Restaurant Count by District")
    result = df.groupBy("district").count().orderBy(F.col("count").desc())
    result.show(truncate=False)
    save_csv(result, "02_district_count")
    return result


def analysis_03_avg_rating_by_cuisine(df):
    section(3, "Average Rating by Cuisine Category")
    result = (
        df.groupBy("cuisine_category")
          .agg(F.round(F.avg("rating"), 3).alias("avg_rating"),
               F.count("*").alias("restaurant_count"))
          .orderBy(F.col("avg_rating").desc())
    )
    result.show(truncate=False)
    save_csv(result, "03_avg_rating_by_cuisine")
    return result


def analysis_04_avg_rating_by_district(df):
    section(4, "Average Rating by District")
    result = (
        df.groupBy("district")
          .agg(F.round(F.avg("rating"), 3).alias("avg_rating"),
               F.count("*").alias("restaurant_count"))
          .orderBy(F.col("avg_rating").desc())
    )
    result.show(truncate=False)
    save_csv(result, "04_avg_rating_by_district")
    return result


def analysis_05_avg_price_by_cuisine(df):
    section(5, "Average Price Level by Cuisine Category")
    result = (
        df.filter(F.col("price_level").isNotNull())
          .groupBy("cuisine_category")
          .agg(F.round(F.avg("price_level"), 2).alias("avg_price_level"))
          .orderBy(F.col("avg_price_level").desc())
    )
    result.show(truncate=False)
    save_csv(result, "05_avg_price_by_cuisine")
    return result


def analysis_06_price_label_distribution(df):
    section(6, "Price Label Distribution")
    result = (
        df.groupBy("price_label")
          .count()
          .orderBy(F.col("count").desc())
    )
    result.show(truncate=False)
    save_csv(result, "06_price_label_distribution")
    return result


def analysis_07_top10_rated(df):
    section(7, "Top 10 Highest Rated Restaurants")
    result = (
        df.orderBy(F.col("rating").desc(), F.col("total_ratings").desc())
          .select("name", "cuisine_category", "district", "rating", "total_ratings", "price_label")
          .limit(10)
    )
    result.show(truncate=False)
    save_csv(result, "07_top10_highest_rated")
    return result


def analysis_08_top10_most_reviewed(df):
    section(8, "Top 10 Most Reviewed Restaurants")
    result = (
        df.orderBy(F.col("total_ratings").desc())
          .select("name", "cuisine_category", "district", "rating", "total_ratings")
          .limit(10)
    )
    result.show(truncate=False)
    save_csv(result, "08_top10_most_reviewed")
    return result


def analysis_09_corr_ratings_total(df):
    section(9, "Correlation: total_ratings vs rating")
    corr = df.stat.corr("total_ratings", "rating")
    print(f"  Pearson correlation (total_ratings ↔ rating): {corr:.4f}")
    result = df.select(
        F.lit(round(corr, 6)).alias("pearson_corr_total_ratings_vs_rating")
    ).limit(1)
    save_csv(result, "09_corr_total_ratings_vs_rating")
    return corr


def analysis_10_corr_price_rating(df):
    section(10, "Correlation: price_level vs rating")
    corr = df.filter(F.col("price_level").isNotNull()).stat.corr("price_level", "rating")
    print(f"  Pearson correlation (price_level ↔ rating): {corr:.4f}")
    result = df.select(
        F.lit(round(corr, 6)).alias("pearson_corr_price_level_vs_rating")
    ).limit(1)
    save_csv(result, "10_corr_price_level_vs_rating")
    return corr


def analysis_11_open_now(df):
    section(11, "Restaurants Open Now vs Closed")
    result = (
        df.groupBy("open_now")
          .count()
          .orderBy(F.col("count").desc())
    )
    result.show(truncate=False)
    save_csv(result, "11_open_now_distribution")
    return result


def analysis_12_popularity_by_district(df):
    section(12, "Average Popularity Score by District")
    result = (
        df.groupBy("district")
          .agg(F.round(F.avg("popularity_score"), 4).alias("avg_popularity_score"),
               F.count("*").alias("count"))
          .orderBy(F.col("avg_popularity_score").desc())
    )
    result.show(truncate=False)
    save_csv(result, "12_popularity_by_district")
    return result


def analysis_13_popularity_by_cuisine(df):
    section(13, "Average Popularity Score by Cuisine Category")
    result = (
        df.groupBy("cuisine_category")
          .agg(F.round(F.avg("popularity_score"), 4).alias("avg_popularity_score"),
               F.count("*").alias("count"))
          .orderBy(F.col("avg_popularity_score").desc())
    )
    result.show(truncate=False)
    save_csv(result, "13_popularity_by_cuisine")
    return result


def analysis_14_rating_buckets(df):
    section(14, "Restaurant Count per Rating Bucket")
    result = (
        df.withColumn(
            "rating_bucket",
            F.when((F.col("rating") >= 1) & (F.col("rating") < 2), "Poor (1–2)")
             .when((F.col("rating") >= 2) & (F.col("rating") < 3), "Average (2–3)")
             .when((F.col("rating") >= 3) & (F.col("rating") < 4), "Good (3–4)")
             .when((F.col("rating") >= 4) & (F.col("rating") <= 5), "Excellent (4–5)")
             .otherwise("Unknown"),
        )
        .groupBy("rating_bucket")
        .count()
        .orderBy("rating_bucket")
    )
    result.show(truncate=False)
    save_csv(result, "14_rating_bucket_distribution")
    return result


def analysis_15_luxury_by_district(df):
    section(15, "Top 5 Districts with Most Luxury Restaurants")
    result = (
        df.filter(F.col("price_label") == "Luxury")
          .groupBy("district")
          .count()
          .orderBy(F.col("count").desc())
          .limit(5)
    )
    result.show(truncate=False)
    save_csv(result, "15_luxury_restaurants_by_district")
    return result


# === MAIN ===

def main() -> None:
    print("=" * 60)
    print("PART A — Step 04: EDA Analytics")
    print("=" * 60)

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

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

    # Explicitly cast numeric columns — inferSchema is unreliable on PySpark-written CSVs
    from pyspark.sql.types import DoubleType, IntegerType
    df = (
        df
        .withColumn("rating",            F.col("rating").cast(DoubleType()))
        .withColumn("total_ratings",     F.col("total_ratings").cast(IntegerType()))
        .withColumn("price_level",       F.col("price_level").cast(IntegerType()))
        .withColumn("review_count",      F.col("review_count").cast(IntegerType()))
        .withColumn("avg_review_rating", F.col("avg_review_rating").cast(DoubleType()))
        .withColumn("popularity_score",  F.col("popularity_score").cast(DoubleType()))
        .withColumn("lat",               F.col("lat").cast(DoubleType()))
        .withColumn("lng",               F.col("lng").cast(DoubleType()))
    )

    df.cache()
    print(f"  Rows loaded: {df.count()}")
    print(f"  Columns: {df.columns}")

    # Run all analyses
    analysis_01_cuisine_count(df)
    analysis_02_district_count(df)
    analysis_03_avg_rating_by_cuisine(df)
    analysis_04_avg_rating_by_district(df)
    analysis_05_avg_price_by_cuisine(df)
    analysis_06_price_label_distribution(df)
    analysis_07_top10_rated(df)
    analysis_08_top10_most_reviewed(df)
    analysis_09_corr_ratings_total(df)
    analysis_10_corr_price_rating(df)
    analysis_11_open_now(df)
    analysis_12_popularity_by_district(df)
    analysis_13_popularity_by_cuisine(df)
    analysis_14_rating_buckets(df)
    analysis_15_luxury_by_district(df)

    spark.stop()
    print(f"\n✅  EDA complete. Results saved to {ANALYTICS_DIR}")


if __name__ == "__main__":
    main()
