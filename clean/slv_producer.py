# Databricks notebook source
import pandas as pd
import re
from event_news_clean import match_admin, merge_event_news, clean_lines, error_handler, process_data
from param_spec import DATABASE_NAME, CLEAN_TABLE

# COMMAND ----------

# Fetch the data
data = match_admin(spark)

# COMMAND ----------

# Drop repetitive columns
data = data.drop(['url_x', 'url_y', 'title_y', 'GLOBALEVENTID_y', 'EventCode_y', 'EventBaseCode_y', 'EventRootCode_y', 'Actor1Code_y', 'Actor1Name_y', 'Actor2Code_y','Actor2Name_y', 'GoldsteinScale_y', 'AvgTone_y' ], axis=1)

# Rename columns ending with '_x'
for name in data.columns:
    if re.search(r'_x$', name):
        new_name = re.sub
        data.rename(columns={name: re.sub(r'_x$', '', name)}, inplace=True)

# COMMAND ----------

# Select the columns in the interests
sample = spark.sql(f'SELECT * FROM openai_gdelt_su_t2.gdelt_news_su_short_viz')
headers = sample.toPandas().columns
headers = list(headers[:-1])
headers.extend(['text', 'title'])
new_data = data.loc[:, headers]

# COMMAND ----------

new_data.info()

# COMMAND ----------

new_data.loc[:, 'title'].fillna('', inplace=True)

# COMMAND ----------

final_data = process_data(new_data, 'text')

# COMMAND ----------

final_data.info()

# COMMAND ----------

final_data

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spdf = spark.createDataFrame(final_data)
spdf.write.mode('append').format('delta').option("mergeSchema", "true").saveAsTable("{}.{}".format(DATABASE_NAME, CLEAN_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Error Analysis

# COMMAND ----------

error_data = clean_lines(data, 'text')
error_data = error_handler(error_data, 'text', drop=False)

# COMMAND ----------

error_data.columns

# COMMAND ----------

error_data.groupby('base')\
        .agg({'empty_return': 'sum', 
              'error_text': 'sum', 
              'error_title': 'sum'})\
        .sort_values(by=['empty_return', 'error_text', 'error_title'], ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### concern

# COMMAND ----------

from param_spec import DATABASE_NAME, ADMIN_TABLE

# COMMAND ----------

admin_df = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ADMIN_TABLE}")
admin_df = admin_df.toPandas()

# COMMAND ----------

admin_df.info()
