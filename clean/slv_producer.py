# Databricks notebook source
import pandas as pd
import re
from event_news_clean import match_admin, merge_event_news, clean_lines, error_handler, process_data
from param_spec import DATABASE_NAME, CLEAN_TABLE, CAMEO_TABLE

# COMMAND ----------

# Fetch the data
data = match_admin(spark)

# COMMAND ----------

# Select the columns in the interests
sample = spark.sql(f'SELECT * FROM openai_gdelt_su_t2.gdelt_news_su_short_viz')
headers = sample.toPandas().columns
headers = list(headers[:-1])
headers.extend(['text', 'title'])
new_data = data.loc[:, headers]
new_data.loc[:, 'title'].fillna('', inplace=True)

# COMMAND ----------

final_data = process_data(new_data, 'text', drop=False)

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spdf = spark.createDataFrame(final_data)
spdf.write.mode('append').format('delta').option("mergeSchema", "true").saveAsTable("{}.{}".format(DATABASE_NAME, CLEAN_TABLE))
