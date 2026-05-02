"""Shared SparkSession factory for all pipeline scripts."""
from pyspark.sql import SparkSession


def build_spark(app_name: str = "ColomboRestaurantPipeline", memory: str = "2g") -> SparkSession:
    """Create (or retrieve) a local SparkSession."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
