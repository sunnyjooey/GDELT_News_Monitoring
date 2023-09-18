# Databricks notebook source
import pandas as pd
import re
from event_news_clean import match_admin, process_data
from param_spec import DATABASE_NAME, CLEAN_TABLE

# COMMAND ----------

# Fetch the data - merge event, title, text, admin together
data = match_admin(spark)
print('Data shape:', data.shape)

# COMMAND ----------

# Select the columns in the interests
headers = ['GLOBALEVENTID', 'DATEADDED', 'SOURCEURL', 'score', 'EventCode','EventBaseCode', 'EventRootCode', 'Actor1Code', 'Actor1Name', 'Actor2Code', 'Actor2Name', 'GoldsteinScale', 'AvgTone', 'Actor1_Adm1', 'Actor2_Adm1', 'Action_Adm1', 'text', 'title']
new_data = data.loc[:, headers]

# COMMAND ----------

# cleaning and finding errors
final_data = process_data(new_data, 'text', drop_error=False)

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spdf = spark.createDataFrame(final_data)
spdf.write.mode('append').format('delta').option("mergeSchema", "true").saveAsTable("{}.{}".format(DATABASE_NAME, CLEAN_TABLE))
