# Databricks notebook source
# !pip install nltk
# !pip install newspaper3k 
# !pip3 install readability-lxml

# COMMAND ----------

# MAGIC %md
# MAGIC #### load libraries 

# COMMAND ----------

# Create Spark Session
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# import libraries
import math
import nltk
import datetime
import warnings
warnings.simplefilter('ignore', FutureWarning)

# import function
from scraper import textgetter
from util import get_last_timestamp
from param_spec import DATABASE_NAME, EVENT_TABLE, ARTICLE_TEXT_TABLE

# COMMAND ----------

# MAGIC %md
# MAGIC #### load data

# COMMAND ----------

# Get last time from article table
start_date = get_last_timestamp(DATABASE_NAME, ARTICLE_TEXT_TABLE, 1, 'dbtimeadded').strftime('%Y-%m-%d') 
# Convert start and end dates to timestamps
start_timestamp = datetime.datetime.strptime(start_date, "%Y-%m-%d")

# get urls from events table
events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EVENT_TABLE}")
events = events.dropDuplicates(['SOURCEURL'])
# so its datetime formate to filter 
events = events.withColumn("DATEADDED", F.to_timestamp("DATEADDED", "yyyyMMddHHmmss"))

# Filter the DataFrame based on the timestamp range
events = events.filter(F.col("DATEADDED") >= F.lit(start_timestamp))

# COMMAND ----------

# MAGIC %md
# MAGIC #### data cleaning

# COMMAND ----------

# add partition by and order by clause if ordering required with in window.
w = Window.orderBy(F.lit(1))
# add row num to manage saving
events = events.withColumn("stop_id", F.row_number().over(w))
events = events.orderBy(F.col("DATEADDED"))
print(events.count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### scraping large df as a job

# COMMAND ----------

idx_date = 0
batch_size_date = 500
total_range = events.count()
num_batches_date = math.ceil(total_range / batch_size_date)
tot_row = 0
now = datetime.datetime.now()

# COMMAND ----------


for batch in range(num_batches_date):
    df = []
    ev = events.filter((events.stop_id > idx_date) & (events.stop_id <= idx_date+batch_size_date))

    for row in ev.collect():
        dct = next(textgetter(row['SOURCEURL']))
        dct.update({'DATEADDED': row['DATEADDED']})
        df.append(dct)

    spdf = spark.createDataFrame(df)
    # add column for keeping track of time added to database
    spdf = spdf.withColumn('dbtimeadded', F.lit(now))
    # write to db
    spdf.write.mode('append').format('delta').option("mergeSchema", "true").saveAsTable("{}.{}".format(DATABASE_NAME, ARTICLE_TEXT_TABLE))
    # tracking
    idx_date += batch_size_date
    num_row = spdf.count()
    tot_row += num_row
    print('Batch', batch, 'has', num_row, 'rows')
    print('Cumulative:', tot_row)
    print('')
    

# COMMAND ----------


