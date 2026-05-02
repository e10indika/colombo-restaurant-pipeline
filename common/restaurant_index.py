"""Shared restaurant StringIndexer utility for Part B scripts."""
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_restaurant_index(df: DataFrame) -> tuple:
    """
    Assign a stable integer restaurant_index to each unique place_id using StringIndexer.

    Returns
    -------
    (df_with_index, indexer_model)
    """
    indexer = StringIndexer(
        inputCol="place_id",
        outputCol="restaurant_index",
        handleInvalid="keep",
    )
    model = indexer.fit(df)
    df = model.transform(df)
    df = df.withColumn("restaurant_index", F.col("restaurant_index").cast("integer"))
    return df, model
