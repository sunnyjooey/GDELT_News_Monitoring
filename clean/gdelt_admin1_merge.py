# Databricks notebook source
!pip install pysal
!pip install descartes
!pip install geopandas

# COMMAND ----------

######## IMPORT PACKAGES #########
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
import functools
from pyspark.sql import DataFrame
from pyspark.sql.functions import to_timestamp, to_date, col, lit, udf
from pyspark.sql.types import IntegerType
from util import get_last_timestamp
from param_spec import COUNTRY_CODES, DATABASE_NAME, EVENT_TABLE, ADMIN_TABLE, SHAPEFILE

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Query Malawi data: 

# COMMAND ----------

# MAGIC %md 
# MAGIC - time period: 'DATEADDED' between 01/02/2022 - 01/08/2023
# MAGIC - country: Code, MI
# MAGIC - actors: actor 1, actor 2, action 2/3 in geocodes 

# COMMAND ----------

# events data
events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EVENT_TABLE}")
#events = events.orderBy("DATEADDED")
#display(events) #is SQL DATE and DATEADDED the same, seems not to be first SQL date is 20190101, 190,01,01 first DATEADDED dates, 2020,01,01.... 

# COMMAND ----------

# inclusive
start_date = get_last_timestamp(DATABASE_NAME, ADMIN_TABLE, 3).strftime('%Y-%m-%d') 
# exclusive
end_date = (get_last_timestamp(DATABASE_NAME, EVENT_TABLE, 3) + dt.timedelta(days=1)).strftime('%Y-%m-%d') 

# Convert start and end dates to timestamps
start_timestamp = datetime.strptime(start_date, "%Y-%m-%d")
end_timestamp = datetime.strptime(end_date, "%Y-%m-%d")

#so its datetime formate to filter 
events = events.withColumn("DATEADDED", to_timestamp("DATEADDED", "yyyyMMddHHmmss"))

# Filter the DataFrame based on the timestamp range
events = events.filter((col("DATEADDED") >= lit(start_timestamp)) & (col("DATEADDED") <= lit(end_timestamp)))
#events.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get admin 1 names

# COMMAND ----------

# smaller table
ev = events.select('GLOBALEVENTID', 'DATEADDED', 'SOURCEURL', 'EventCode', 'EventBaseCode', 'EventRootCode', 'Actor1Code', 'Actor1Name', 'Actor2Code', 'Actor2Name', 'GoldsteinScale', 'AvgTone', 'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor2Geo_Lat', 'Actor2Geo_Long', 'ActionGeo_Lat', 'ActionGeo_Long')
evp = ev.toPandas()

# COMMAND ----------

import geopandas as gpd
gdf = gpd.read_file(SHAPEFILE)

# COMMAND ----------

# change lat-lon to admin 1 names
from shapely.geometry import Point
geometry1 = [Point(xy)  for xy in zip(evp['Actor1Geo_Long'], evp['Actor1Geo_Lat'])]
evp = gpd.GeoDataFrame(evp, crs=gdf.crs, geometry=geometry1)
evp = gpd.sjoin(evp, gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
evp = evp.rename(columns={'ADM1_EN': 'Actor1_Adm1'})
evp = evp.drop(['Actor1Geo_Long', 'Actor1Geo_Lat', 'Shape_Leng', 'Shape_Area', 'ADM1_PCODE', 'ADM0_EN', 'ADM0_PCODE', 'date', 'validOn', 'validTo', 'geometry', 'index_right'], axis=1)

# COMMAND ----------

geometry2 = [Point(xy)  for xy in zip(evp['Actor2Geo_Long'], evp['Actor2Geo_Lat'])]
evp = gpd.GeoDataFrame(evp, crs=gdf.crs, geometry=geometry2)
evp = gpd.sjoin(evp, gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
evp = evp.rename(columns={'ADM1_EN': 'Actor2_Adm1'})
evp = evp.drop(['Actor2Geo_Long', 'Actor2Geo_Lat', 'Shape_Leng', 'Shape_Area', 'ADM1_PCODE', 'ADM0_EN', 'ADM0_PCODE', 'date', 'validOn', 'validTo', 'geometry', 'index_right'], axis=1)

# COMMAND ----------

geometry3 = [Point(xy)  for xy in zip(evp['ActionGeo_Long'], evp['ActionGeo_Lat'])]
evp = gpd.GeoDataFrame(evp, crs=gdf.crs, geometry=geometry3)
evp = gpd.sjoin(evp, gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
evp = evp.rename(columns={'ADM1_EN': 'Action_Adm1'})
evp = evp.drop(['ActionGeo_Long', 'ActionGeo_Lat', 'Shape_Leng', 'Shape_Area', 'ADM1_PCODE', 'ADM0_EN', 'ADM0_PCODE', 'date', 'validOn', 'validTo', 'geometry', 'index_right'], axis=1)

# COMMAND ----------

evp = evp.loc[:, ['GLOBALEVENTID', 'Actor1_Adm1','Actor2_Adm1','Action_Adm1']]

# COMMAND ----------

# fill in nan with country codes
evp[['Actor1_Adm1','Actor2_Adm1','Action_Adm1']] = evp[['Actor1_Adm1','Actor2_Adm1','Action_Adm1']].fillna(COUNTRY_CODES)

# COMMAND ----------

# change to spark df
# Note: Make sure the DBR version used on the cluser is >= 13.x
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
evp_sdf = spark.createDataFrame(evp)

# COMMAND ----------

# write save to pyspark
evp_sdf.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, ADMIN_TABLE))
