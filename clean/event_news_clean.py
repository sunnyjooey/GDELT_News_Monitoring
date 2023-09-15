# import libraries
import datetime as dt
import os
import re
import pandas as pd
import warnings
warnings.simplefilter('ignore', FutureWarning)
from param_spec import EVENT_TABLE, EMBED_TABLE, ARTICLE_TEXT_TABLE, CLEAN_TABLE, DATABASE_NAME, ADMIN_TABLE, TARGET_CAMEO, COUNTRY_CODES
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType


def create_score_col(spark_event_df):
    """
    Create a score column for the event table
    """

    def score(act1, act2, action):
        return sum([act1==COUNTRY_CODES, act2==COUNTRY_CODES, action==COUNTRY_CODES])
    score_udf = F.udf(score, IntegerType())

    spark_event_df = spark_event_df.withColumn('score', score_udf(spark_event_df.Actor1Geo_CountryCode, spark_event_df.Actor2Geo_CountryCode, spark_event_df.ActionGeo_CountryCode))

    return spark_event_df


def merge_event_news(spark):
    """
    Merge the event and news data by their common url

    Note: "spark" is pre-configured in notebooks through databricks so this func can be directly run in databricks notebooks cell.
    """

    # for filtering events data for easier merging
    events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EVENT_TABLE}")
    events = events.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
    events = events.withColumn('DATEADDED', F.to_date('DATEADDED'))
    events = events.filter((events.IsRootEvent == '1')) # Filter to root events
    events = events.filter(events.EventRootCode.isin(TARGET_CAMEO)) # Filter to target CAMEOs
    events = create_score_col(events) # create score column
    events = events.filter(events.score >= 2) # Filter by assigned score
    events = events.dropDuplicates(['SOURCEURL'])
    events = events.toPandas()

    # for filtering titles data for easier merging
    titles = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EMBED_TABLE}")
    titles = titles.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
    titles = titles.withColumn('DATEADDED', F.to_date('DATEADDED'))
    titles = titles.dropDuplicates(['url'])
    titles = titles.toPandas()
    titles.rename(columns={'DATEADDED': 'DATEADDED_titles'}, inplace=True)

    event_title = pd.merge(events, titles, 'left', left_on='SOURCEURL', right_on='url')

    # for filtering texts data for easier merging
    texts =  spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ARTICLE_TEXT_TABLE}")
    texts = texts.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
    texts = texts.withColumn('DATEADDED', F.to_date('DATEADDED'))
    texts = texts.dropDuplicates(['url'])
    texts = texts.toPandas()
    texts.rename(columns={'DATEADDED': 'DATEADDED_texts'}, inplace=True)

    return pd.merge(event_title, texts, 'left', left_on='SOURCEURL', right_on='url')


def match_admin(spark):
    """
    Match the event-news data with the corresponding admin levels
    """

    admins = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ADMIN_TABLE}")
    admins = admins.toPandas()
    event_news = merge_event_news(spark)
    event_admin_news = pd.merge(event_news, admins, 'left', on='GLOBALEVENTID')
    # Drop repetitive columns
    event_admin_news = event_admin_news.drop(['url_x', 'url_y', 'title_y'], axis=1)

    # Rename columns ending with '_x'
    for name in event_admin_news.columns:
        if re.search(r'_x$', name):
            event_admin_news.rename(columns={name: re.sub(r'_x$', '', name)}, inplace=True)

    return event_admin_news


def clean_lines(df, text_col):
    """
    Clean new lines, spaces
    Args:
        df: the dataframe containing texts
        text_col: the text column to clean
    """

    article_df = df.copy()
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
    
    return article_df


def error_handler(df, text_col, drop=True):
    """
    Detect/drop rows with error messages
    Args:
        df: dataframe
        text_col: the text col
        drop: bool specifying whether to drop the flag columns and rows with errors
    """

    article_df = df.copy()
    text_col_clean = f'{text_col}_clean'

    # Flag empty values
    article_df.loc[:,'empty_return'] = 0
    article_df.loc[(article_df[text_col_clean] == '') |
                   (article_df[text_col_clean] == 'None'),'empty_return'] = 1

    # take out known error messages - text col (hard code)
    article_df[text_col_clean].fillna('', inplace=True)
    article_df['error_text'] = article_df[text_col_clean].apply(lambda x: 1 if re.search(r'(something went wrong, please try again later)|(cloudflare ray)|(legal disclaimer)|(page unavailable)|(website is using a security service to protect itself from online attacks)|(is using a security service for protection against online attacks)|(412 error)|(access denied - godaddy website)', x, re.IGNORECASE) else 0)

    # take out known error messages - title col (hard code)
    article_df['title'].fillna('', inplace=True)
    article_df['error_title'] = article_df['title'].apply(lambda x: 1 if re.search(r'(page not found)|(are you a robot)|porn|biztoc', x, re.IGNORECASE) else 0)

    if drop:
        article_df = article_df[article_df['error_text']==0]
        article_df = article_df.drop('error_text', axis=1)
        article_df = article_df[article_df['error_title']==0]
        article_df = article_df.drop('error_title', axis=1)
        article_df = article_df[article_df['empty_return']==0]
        article_df = article_df.drop('empty_return', axis=1)
    
    return article_df


def process_data(ord_data, text_col, drop=True):
    """
    Process and clean the article DataFrame.
    Saves processed data as the attribute processed_df
    Args:
        ord_data: the dataframe containing texts
        text_col: the text column to clean
        drop: bool specifying whether to drop the flag columns and rows with errors
    """

    article_df = ord_data.copy()
    print(f'Original number of articles: {article_df.shape[0]}')

    # clean new lines, spaces
    artical_df = clean_lines(article_df, text_col)

    # take out known error messages
    article_df = error_handler(artical_df, text_col, drop)

    # Save the processed DataFrame as processed_df
    print(f'Number of articles after cleaning: {article_df.shape[0]}')

    return article_df.copy()
