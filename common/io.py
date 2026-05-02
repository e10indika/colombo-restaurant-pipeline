"""Shared I/O helpers for the Colombo Restaurant pipeline."""
import sys
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame


def load_csv(spark: SparkSession, path: str, *, infer_schema: bool = True) -> DataFrame:
    """
    Load a CSV (file or PySpark directory) into a DataFrame.
    Raises SystemExit(1) on failure so callers don't need try/except.
    """
    try:
        p = Path(path)
        # PySpark writes a directory of part-*.csv files — find the data file
        if p.is_dir():
            parts = sorted(p.glob("part-*.csv"))
            if not parts:
                raise FileNotFoundError(f"No part-*.csv files found in {path}")
            read_path = str(parts[0])
        else:
            read_path = path

        return (
            spark.read
            .option("header", "true")
            .option("inferSchema", str(infer_schema).lower())
            .csv(read_path)
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load CSV from {path}: {exc}", file=sys.stderr)
        spark.stop()
        sys.exit(1)


def save_csv(df: DataFrame, path: str, *, coalesce: int = 1) -> None:
    """Write a DataFrame as a single-partition CSV (overwrites existing)."""
    (
        df.coalesce(coalesce)
        .write
        .option("header", "true")
        .mode("overwrite")
        .csv(path)
    )
