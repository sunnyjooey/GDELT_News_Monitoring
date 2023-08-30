# import libraries
import datetime as dt
import os
import pandas as pd
import warnings
warnings.simplefilter('ignore', FutureWarning)
from param_spec import INPUT_TABLE_NAME, OUTPUT_TABLE_NAME, DATABASE_NAME
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType


def merge_event_news(spark):
    """
    Merge the event and news data by their common url

    Note: "spark" is pre-configured in notebooks through databricks so this func can be directly run in databricks notebooks cell.
    """

    # for filtering events data for easier merging
    events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{INPUT_TABLE_NAME}")
    events = events.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
    events = events.withColumn('DATEADDED', F.to_date('DATEADDED'))
    events = events.filter((events.IsRootEvent == '1'))
    events = events.dropDuplicates(['SOURCEURL'])
    events = events.toPandas()

    # for filtering news data for easier merging
    news = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{OUTPUT_TABLE_NAME}")
    news = news.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
    news = news.withColumn('DATEADDED', F.to_date('DATEADDED'))
    news = news.dropDuplicates(['url'])
    news = news.toPandas()
    news.rename(columns={'DATEADDED': 'DATEADDED_news'}, inplace=True)

    return pd.merge(events, news, 'inner', left_on='SOURCEURL', right_on='url')
