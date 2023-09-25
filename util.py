import pandas as pd
import datetime as dt
import re
from pyspark.sql.types import StructField, BooleanType, StringType, IntegerType, FloatType, ArrayType, TimestampType
from pyspark.sql import SparkSession
from param_spec import ARTICLE_TEXT_TABLE

# Create SparkSession
spark = SparkSession.builder.getOrCreate()


############# Common Functions #############
def get_last_timestamp(db_name, table_name, w, date_col='DATEADDED'):
    # If table is new, go w weeks back
    tables = [table.name for table in spark.catalog.listTables(dbName=db_name)]
    if table_name in tables:
        # list search - must be more than one day from now to account for error handling
        d = dt.datetime.now() - dt.timedelta(days=1)

        if table_name == ARTICLE_TEXT_TABLE:
            # transform date format for ARTICLE_TEXT_TABLE
            last_ts = spark.sql(f"""SELECT MAX(ts_time) AS last_timestamp FROM (SELECT date_format(from_unixtime(unix_timestamp({date_col}, "yyyy-MM-dd'T'HH:mm:ss.SSSZ")), "yyyyMMddHHmmss") AS ts_time FROM {db_name}.{table_name}) WHERE ts_time < '{d.year}{d.month}{d.day}{d.hour}{d.minute}{d.second}'""")
        else:
            last_ts = spark.sql(f"SELECT MAX({date_col}) as last_timestamp FROM {db_name}.{table_name} WHERE {date_col} < '{d.year}{d.month}{d.day}{d.hour}{d.minute}{d.second}'")
            
        last_ts = last_ts.first()['last_timestamp']
        w_wk_ago = pd.to_datetime((dt.datetime.now() - dt.timedelta(weeks=w)), format='%Y%m%d%H%M%S')
        #w_wk_ago = pd.to_datetime((dt.datetime.now() - dt.timedelta(days=w)), format='%Y%m%d%H%M%S')
        
        if last_ts is None:
            ret = w_wk_ago
        else:
            last_ts = pd.to_datetime(last_ts, format='%Y%m%d%H%M%S')
            # return the later time of the two
            ret = max(w_wk_ago, last_ts) 
    else:
        ret = pd.to_datetime((dt.datetime.now() - dt.timedelta(weeks=w)), format='%Y%m%d%H%M%S')
        #ret = pd.to_datetime((dt.datetime.now() - dt.timedelta(days=w)), format='%Y%m%d%H%M%S')

    return ret
