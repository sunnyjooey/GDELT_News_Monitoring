# import libraries
import datetime as dt
import os
import pandas as pd
import warnings
warnings.simplefilter('ignore', FutureWarning)
from param_spec import EVENT_TABLE, EMBED_TABLE, ARTICLE_TEXT_TABLE, CLEAN_TABLE, DATABASE_NAME, ADMIN_TABLE
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType


def merge_event_news(spark):
    """
    Merge the event and news data by their common url

    Note: "spark" is pre-configured in notebooks through databricks so this func can be directly run in databricks notebooks cell.
    """

    # for filtering events data for easier merging
    events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EVENT_TABLE}")
    events = events.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
    events = events.withColumn('DATEADDED', F.to_date('DATEADDED'))
    events = events.filter((events.IsRootEvent == '1'))
    events = events.dropDuplicates(['SOURCEURL'])
    events = events.toPandas()

    # for filtering titles data for easier merging
    titles = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EMBED_TABLE}")
    titles = titles.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
    titles = titles.withColumn('DATEADDED', F.to_date('DATEADDED'))
    titles = titles.dropDuplicates(['url'])
    titles = titles.toPandas()
    titles.rename(columns={'DATEADDED': 'DATEADDED_titles'}, inplace=True)

    event_title = pd.merge(events, titles, 'inner', left_on='SOURCEURL', right_on='url')

    # for filtering texts data for easier merging
    texts =  spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ARTICLE_TEXT_TABLE}")
    texts = texts.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
    texts = texts.withColumn('DATEADDED', F.to_date('DATEADDED'))
    texts = texts.dropDuplicates(['url'])
    texts = texts.toPandas()
    texts.rename(columns={'DATEADDED': 'DATEADDED_texts'}, inplace=True)

    return pd.merge(event_title, texts, 'inner', left_on='SOURCEURL', right_on='url')


def match_admin(spark):
    """
    Match the event-news data with the corresponding admin levels
    """

    admins = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ADMIN_TABLE}")
    admins = admins.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
    admins = admins.withColumn('DATEADDED', F.to_date('DATEADDED'))
    admins = admins.dropDuplicates(['SOURCEURL'])
    admins = admins.toPandas()
    admins.rename(columns={'DATEADDED': 'DATEADDED_admin'}, inplace=True)

    event_news = merge_event_news(spark)

    return pd.merge(event_news, admins, 'inner', on='SOURCEURL')


def process_data(ord_data, text_col):
    """
    Process and clean the article DataFrame.
    Saves processed data as the attribute processed_df
    Args:
        ord_data: the dataframe containing texts
        text_col: the text column to clean
    """

    article_df = ord_data.copy()
    print(f'Original number of articles: {article_df.shape[0]}')

    # clean new lines, spaces 
    text_col_clean = f'{text_col}_clean'
    article_df[text_col_clean] = article_df[text_col].astype(str)
    article_df[text_col_clean] = article_df[text_col_clean].apply(lambda x: x.replace('\s', ' ').strip())
    article_df[text_col_clean] = article_df[text_col_clean].apply(lambda x: x.replace('\n', ' ').strip())  # Remove newlines
    article_df[text_col_clean] = article_df[text_col_clean].apply(lambda x: x.replace(' - ', ' ').strip())  # Remove hyphens
    article_df[text_col_clean] = article_df[text_col_clean].apply(lambda x: x.replace(' -- ', ' ').strip())
    article_df[text_col_clean] = article_df[text_col_clean].apply(lambda x: x.replace('…', ' ').strip())
    article_df[text_col_clean] = article_df[text_col_clean].apply(lambda x: x.replace('“', '"').strip())
    article_df[text_col_clean] = article_df[text_col_clean].apply(lambda x: x.replace('”', '"'.strip()))
    article_df[text_col_clean] = article_df[text_col_clean].apply(lambda x: re.sub('\s\s+' , ' ', x)) # Condense multiple spaces to one
    article_df = article_df[article_df[text_col_clean] != '']
    article_df = article_df[article_df[text_col_clean] != 'None']
    
    # take out known error messages
    article_df['bad'] = article_df[text_col_clean].apply(lambda x: 1 if re.search(r'(something went wrong, please try again later)|(cloudflare ray)|(legal disclaimer)|(page unavailable)|(website is using a security service to protect itself from online attacks)|(is using a security service for protection against online attacks)|(412 error)|(access denied - godaddy website)', x, re.IGNORECASE) else 0)
    article_df = article_df[article_df['bad']==0]
    article_df = article_df.drop('bad', axis=1)
    # take out known error messages - title col (hard code)
    article_df['bad'] = article_df['title'].apply(lambda x: 1 if re.search(r'(page not found)|(are you a robot)|porn|biztoc', x, re.IGNORECASE) else 0)
    article_df = article_df[article_df['bad']==0]
    article_df = article_df.drop('bad', axis=1)

    # Save the processed DataFrame as processed_df
    print(f'Number of articles after cleaning: {article_df.shape[0]}')

    return article_df.copy()
