# Databricks notebook source
!pip install pysal
!pip install descartes
!pip install geopandas

# COMMAND ----------

######## IMPORT PACKAGES #########
import numpy as np
import pandas as pd
from datetime import datetime
import functools
from pyspark.sql import DataFrame
from pyspark.sql.functions import to_timestamp, to_date, col, lit, udf
from pyspark.sql.types import IntegerType
from param_spec import COUNTRY_CODES, DATABASE_NAME, EVENT_TABLE, ADMIN_TABLE, START_DATE, END_DATE

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
events = events.orderBy("DATEADDED")
display(events) #is SQL DATE and DATEADDED the same, seems not to be first SQL date is 20190101, 190,01,01 first DATEADDED dates, 2020,01,01.... 

# COMMAND ----------

# filter to root events only 
events = events.filter(events.IsRootEvent=='1')
events.count()

# COMMAND ----------

start_date = START_DATE
end_date = END_DATE

# Convert start and end dates to timestamps
start_timestamp = datetime.strptime(start_date, "%Y-%m-%d")
end_timestamp = datetime.strptime(end_date, "%Y-%m-%d")

#so its datetime formate to filter 
events = events.withColumn("DATEADDED", to_timestamp("DATEADDED", "yyyyMMddHHmmss"))

# Filter the DataFrame based on the timestamp range
events = events.filter((col("DATEADDED") >= lit(start_timestamp)) & (col("DATEADDED") <= lit(end_timestamp)))
events.count()

# COMMAND ----------

# count how many are in MI (0 - 3)
def score(act1, act2, action):
    return sum([act1==COUNTRY_CODES, act2==COUNTRY_CODES, action==COUNTRY_CODES])
score_udf = udf(score, IntegerType())

events = events.withColumn('score', score_udf(events.Actor1Geo_CountryCode, events.Actor2Geo_CountryCode, events.ActionGeo_CountryCode))

# COMMAND ----------

# filter to events in sudan (2 or more)
events = events.filter(events.score >= 2)
events.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Theme digging: using CAMEO code 

# COMMAND ----------

# MAGIC %md
# MAGIC filter by CAMEO Codes of interest: 
# MAGIC - 60% of recorded political violence incidents in Khartoum 
# MAGIC - violence agaisnt civilians tend to be high (2023)
# MAGIC - prior year before darfur was the area of high pol violence
# MAGIC - protests are key/during anniverarys 
# MAGIC - themes: Protests (14), Exhibit force posture (15), coerce (17), assulat (18), fight(19)

# COMMAND ----------

# filter event codes
# note that URLs are not unique!
events_conflict = events.filter(events.EventRootCode.isin(['14','15','17','18','19']))
display(events_conflict.groupBy('EventRootCode').count())
#Protests (14), Exhibit force posture (15), coerce (17), assulat (18), fight(19)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Get admin 1 names

# COMMAND ----------

# smaller table
ev = events_conflict.select('GLOBALEVENTID', 'DATEADDED', 'SOURCEURL', 'score', 'EventCode', 'EventBaseCode', 'EventRootCode', 'Actor1Code', 'Actor1Name', 'Actor2Code', 'Actor2Name', 'GoldsteinScale', 'AvgTone', 'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor2Geo_Lat', 'Actor2Geo_Long', 'ActionGeo_Lat', 'ActionGeo_Long')
evp = ev.toPandas()

# COMMAND ----------

import geopandas as gpd
SHAPEFILE = '/dbfs/FileStore/df/shapefiles/malawi_admin1'
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

evp.info()

# COMMAND ----------

geometry2 = [Point(xy)  for xy in zip(evp['Actor2Geo_Long'], evp['Actor2Geo_Lat'])]
evp = gpd.GeoDataFrame(evp, crs=gdf.crs, geometry=geometry2)
evp = gpd.sjoin(evp, gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
evp = evp.rename(columns={'ADM1_EN': 'Actor2_Adm1'})
evp = evp.drop(['Actor2Geo_Long', 'Actor2Geo_Lat', 'Shape_Leng', 'Shape_Area', 'ADM1_PCODE', 'ADM0_EN', 'ADM0_PCODE', 'date', 'validOn', 'validTo', 'geometry', 'index_right'], axis=1)

# COMMAND ----------

evp.info()

# COMMAND ----------

geometry3 = [Point(xy)  for xy in zip(evp['ActionGeo_Long'], evp['ActionGeo_Lat'])]
evp = gpd.GeoDataFrame(evp, crs=gdf.crs, geometry=geometry3)
evp = gpd.sjoin(evp, gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
evp = evp.rename(columns={'ADM1_EN': 'Action_Adm1'})
evp = evp.drop(['ActionGeo_Long', 'ActionGeo_Lat', 'Shape_Leng', 'Shape_Area', 'ADM1_PCODE', 'ADM0_EN', 'ADM0_PCODE', 'date', 'validOn', 'validTo', 'geometry', 'index_right'], axis=1)

# COMMAND ----------

evp.info()

# COMMAND ----------

# fill in nan with MI
evp[['Actor1_Adm1','Actor2_Adm1','Action_Adm1']] = evp[['Actor1_Adm1','Actor2_Adm1','Action_Adm1']].fillna('MI')

# COMMAND ----------

evp.head(5)

# COMMAND ----------



# COMMAND ----------

# change to spark df
# Note: Make sure the DBR version used on the cluser is >= 13.x
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
evp_sdf = spark.createDataFrame(evp)

# COMMAND ----------

# write save to pyspark
evp_sdf.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, ADMIN_TABLE))
