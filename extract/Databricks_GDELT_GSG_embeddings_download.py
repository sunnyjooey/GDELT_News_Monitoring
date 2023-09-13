# Databricks notebook source
# import libraries
import math
import numpy as np
import datetime as dt
import re
import os
import pandas as pd
import json
import urllib
import gzip
import gc
from io import BytesIO
import warnings
warnings.simplefilter('ignore', FutureWarning)
from param_spec import EVENT_TABLE, EMBED_TABLE, ERROR_TABLE, DATABASE_NAME
from util import get_last_timestamp
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# COMMAND ----------

# Input Params
# inclusive
start_date = get_last_timestamp(DATABASE_NAME, EMBED_TABLE, 1).strftime('%Y-%m-%d') 
# exclusive
end_date = (get_last_timestamp(DATABASE_NAME, EVENT_TABLE, 1) + dt.timedelta(days=1)).strftime('%Y-%m-%d')

# COMMAND ----------

# define search range (15 min chunks in each day)
def get_date_time_intervals(_start_date, _end_date):
    _date_time_range = []
    _range = pd.date_range(start=_start_date, end=_end_date, freq='15T')
    for _date in _range:
        _date_time_range.append(str(_date))
    return _date_time_range

# get dates
date_range = get_date_time_intervals(start_date, end_date)

# exclude last time frame published at midnight for last 15 min from day before
date_range = date_range[:-1]
print('Number of time intervals:', len(date_range))

# COMMAND ----------

# define schema 
str_cols = ['date', 'url', 'lang', 'title']
str_schema = [StructField(col, StringType(), True) for col in str_cols]
str_schema.extend([StructField('DATEADDED', StringType(), True)])
# error df schema
eschema = [StructField('date', StringType(), True), StructField('data', StringType(), True), StructField('error', StringType(), True)]

# COMMAND ----------

# get data from GSG in batches
idx_date = 0
batch_size_date = 96
total_range = len(date_range)
num_batches_date = math.ceil(total_range / batch_size_date)

for batch in range(num_batches_date):
    print('Batch:', batch)
    # instatiate df for batch of news
    _gdelt_data_batch = pd.DataFrame()

    # window of a few days before and after
    day = date_range[idx_date]
    day = dt.datetime.strptime(day, '%Y-%m-%d %H:%M:%S').date()
    before = day + dt.timedelta(-3)
    after = day + dt.timedelta(2)

    # for filtering events data for easier merging
    events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EVENT_TABLE}")
    events = events.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
    events = events.withColumn('DATEADDED', F.to_date('DATEADDED'))
    events = events.filter((events.DATEADDED >= before) & (events.DATEADDED <= after))
    events = events.dropDuplicates(['SOURCEURL'])
    events = events.toPandas()

    for count, date in enumerate(date_range[idx_date:idx_date+batch_size_date]):
        if count % 48 == 0:
            print(date)
        _date = re.sub(r'[^\d+]','', date)
        
        try:
            resp = urllib.request.urlopen(f'http://data.gdeltproject.org/gdeltv3/gsg_docembed/{_date}.gsg.docembed.json.gz')
            # unzip compressed file
            with gzip.open(BytesIO(resp.read())) as f:
                _gdelt_data = f.read()
            _gdelt_data = _gdelt_data.decode("utf-8") 
            _gdelt_data = _gdelt_data.split('\n')
            _gdelt_data = [json.loads(f) for f in _gdelt_data if len(f)>0]
            _gdelt_data = pd.DataFrame(_gdelt_data)
            _gdelt_data.drop(inplace=True, columns=['model', 'docembed'])

            # merge with events data to save only relevant rows
            merged = pd.merge(events, _gdelt_data, left_on='SOURCEURL', right_on='url')
            _gdelt_data = merged.loc[: , _gdelt_data.columns].copy()
            _gdelt_data['DATEADDED'] = _date
            # append news to df for batch
            _gdelt_data_batch = pd.concat([_gdelt_data_batch, _gdelt_data])
            # clear from memory
            del _gdelt_data

        except Exception as e:
            edf = pd.DataFrame({'date':[_date], 'data':['gsg_embed'], 'error':[str(e)]})
            spedf = spark.createDataFrame(edf, StructType(eschema))
            spedf.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, ERROR_TABLE))
            print(f'#### FAILED AT {date} - ####')

    # reset index
    _gdelt_data_batch.reset_index(inplace=True, drop=True)
    print('ALL GSG embeddings in batch:', _gdelt_data_batch.shape)
    
    # convert to spark
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spdf = spark.createDataFrame(_gdelt_data_batch, StructType(str_schema))
    # save output
    spdf.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, EMBED_TABLE))

    # unpdate indices for next batch
    idx_date += batch_size_date

    # clear batch variables from memory
    del _gdelt_data_batch


# COMMAND ----------


