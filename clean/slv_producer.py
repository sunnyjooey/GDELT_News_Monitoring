# Databricks notebook source
import pandas as pd
import re
from event_news_clean import match_admin, merge_event_news, process_data
from param_spec import DATABASE_NAME, CLEAN_TABLE

# COMMAND ----------

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

data.head(3)

# COMMAND ----------

data.info()

# COMMAND ----------

data = process_data(data, 'text')

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spdf = spark.createDataFrame(data)
spdf.write.mode('append').format('delta').option("mergeSchema", "true").saveAsTable("{}.{}".format(DATABASE_NAME, CLEAN_TABLE))

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
