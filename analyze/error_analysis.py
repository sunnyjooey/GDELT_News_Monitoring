# Databricks notebook source
import pandas as pd
from clean.event_news_clean import match_admin, clean_lines, error_handler
from param_spec import CLEAN_TABLE, DATABASE_NAME

# COMMAND ----------

# Fetch the data
data = match_admin(spark)
error_data = clean_lines(data, 'text')
error_data = error_handler(error_data, 'text', drop=False)

# COMMAND ----------

error_data.groupby('base')\
        .agg({'empty_return': 'sum', 
              'error_text': 'sum', 
              'error_title': 'sum'})\
        .sort_values(by=['empty_return', 'error_text', 'error_title'], ascending=False)
