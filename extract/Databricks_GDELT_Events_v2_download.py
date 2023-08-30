# Databricks notebook source
# import libraries
import math
import re
import os
import pandas as pd
import numpy as np
import datetime as dt
import urllib
import zipfile
import gc
from io import BytesIO
from param_spec import START_DATE, END_DATE, COUNTRY_CODES, DATABASE_NAME, TABLE_NAME, ERROR_TABLE_NAME
import warnings
warnings.simplefilter('ignore', FutureWarning)

from pyspark.sql.types import StructType, StructField, StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ### Updates:
# MAGIC 1. **Updated the country code for Malawi** \
# MAGIC Code book: Encoded based on FIPS country codes \
# MAGIC FIPS Country code source: https://en.wikipedia.org/wiki/List_of_FIPS_country_codes
# MAGIC
# MAGIC 2. **Code Modification: pd.concat V.S. df.append**\
# MAGIC Same function but when working with large datasets, pd.concat will be more efficient since Append function will add rows of second data frame to first dataframe iteratively one by one. Concat function will do a single operation to finish the job, which makes it faster than append\
# MAGIC https://medium.com/analytics-vidhya/a-tip-a-day-python-tip-5-pandas-concat-append-dev-skrol-18e4950cc8cc
# MAGIC
# MAGIC 3. **Sample data saved as a delta table**

# COMMAND ----------

a = "MI"
a.split(',')

# COMMAND ----------


# Input Params
start_date = START_DATE  # inclusive
end_date = END_DATE  # exclusive: download does not include this day 
country_codes = COUNTRY_CODES.split(',')


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

# define GDELT schema
gdelt_event_url = "https://raw.githubusercontent.com/linwoodc3/gdelt2HeaderRows/master/schema_csvs/GDELT_2.0_Events_Column_Labels_Header_Row_Sep2016.csv"
gdelt_events_schema = pd.read_csv(gdelt_event_url, sep=',')
gdelt_events_headers = gdelt_events_schema['tableId'].values

# COMMAND ----------

# get data from Events 2.0 in batches

idx_date = 0
batch_size_date = 96
total_range = len(date_range)
num_batches_date = math.ceil(total_range / batch_size_date) # num of the batches
gdelt_data_full_search = pd.DataFrame()
error_df = pd.DataFrame()

for batch in range(num_batches_date):
    print('Batch:', batch)
    # instatiate df for batch of news
    _gdelt_data_batch = pd.DataFrame()

    for count, date in enumerate(date_range[idx_date:idx_date+batch_size_date]):
        if count % 48 == 0:
            print(date)
        _date = re.sub(r'[^\d+]','', date)
        try:
            # fetch gdelt events data
            _gdelt_data = pd.read_csv(f"http://data.gdeltproject.org/gdeltv2/{_date}.export.CSV.zip", 
                                      lineterminator='\n', delimiter='\t', encoding='utf8', header=None, compression='zip')
            _gdelt_data.columns = gdelt_events_headers
            # remove undated rows
            _gdelt_data = _gdelt_data.dropna(subset=['SQLDATE']) 
            # append news to df for batch
            _gdelt_data_batch = pd.concat([_gdelt_data_batch, _gdelt_data])
        except Exception as e:
            edf = pd.DataFrame({'date':[_date], 'data':['events2'], 'error':[str(e)]})
            error_df = pd.concat([error_df, edf])
            print(f'#### FAILED AT {date} ####')

    # reset index
    _gdelt_data_batch.reset_index(inplace=True, drop=True)
    # shape of unmodified events from batch
    print('ALL GDELT 2.0 events in batch:', _gdelt_data_batch.shape)
    # select data from events for defined country 
    # note: we are not filtering by Actor1CountryCode (and 2) because they do not seem to be accurate
    _gdelt_data_batch = _gdelt_data_batch.loc[(_gdelt_data_batch.ActionGeo_CountryCode.isin(country_codes)) | 
                                              (_gdelt_data_batch.Actor1Geo_CountryCode.isin(country_codes)) | 
                                              (_gdelt_data_batch.Actor2Geo_CountryCode.isin(country_codes))].copy()
    print('Number of country relevant events:', _gdelt_data_batch.shape[0])
    # reset index
    _gdelt_data_batch.reset_index(inplace=True, drop=True)
    # append batch to full search results df
    gdelt_data_full_search = pd.concat([gdelt_data_full_search, _gdelt_data_batch])
    # unpdate indices for next batch
    idx_date += batch_size_date

    # clear batch variables from memory
    del _gdelt_data_batch
    gc.collect()
    del _gdelt_data
    gc.collect()
    
gdelt_data_full_search.reset_index(inplace=True, drop=True)

# COMMAND ----------

# minor cleaning - drop columns
gdelt_data_full_search = gdelt_data_full_search.drop(columns=['MonthYear', 'Year', 'FractionDate'])

# COMMAND ----------

# define schema -- data is messy, so cast everything as string for now
schema = [StructField(col, StringType(), True) for col in gdelt_data_full_search.columns]

# COMMAND ----------

# convert to spark
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spdf = spark.createDataFrame(gdelt_data_full_search, StructType(schema))

# COMMAND ----------

# save output
spdf.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, TABLE_NAME))

# COMMAND ----------

# save error table if any
if error_df.shape[0] > 0:
    eschema = [StructField(col, StringType(), True) for col in error_df.columns]
    spedf = spark.createDataFrame(error_df, StructType(eschema))
    spedf.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, ERROR_TABLE_NAME))
