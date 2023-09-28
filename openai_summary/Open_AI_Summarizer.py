# Databricks notebook source
# MAGIC %md
# MAGIC ## TEXT SUMMARISATION OF GDELT NEWS TEXT USING GPT3 MODEL

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Packages, Credentials & Load Data

# COMMAND ----------

!pip install openai python-dotenv
!pip install azure-identity

# COMMAND ----------

#### packages ####
import openai
from summariser import Summariser
from param_spec import DATABASE_NAME, CLEAN_TABLE, SUMMARY_TABLE, ARG_TABLE
import pandas as pd

# COMMAND ----------

#### creds ######
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

openai_key = dbutils.secrets.get(scope='openai_scope', key='openai_key')
openai_endpoint = dbutils.secrets.get(scope='openai_scope', key='openai_endpoint')

# COMMAND ----------

# API CREDS WORKS
openai.api_key = openai_key
# your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_base = openai_endpoint 
openai.api_type = 'azure'
# this may change in the future
openai.api_version = "2022-12-01" 
# This will correspond to the custom name you chose for your deployment when you deployed a model. 
deployment_name = 'gdelt_test' 

# COMMAND ----------

# Send a completion call to generate an answer, TEST for API working 
print('Sending a test completion job')
start_phrase = 'Write a tagline for a flower shop. '
response = openai.Completion.create(engine=deployment_name, prompt=start_phrase, max_tokens=10)
text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
print(start_phrase+text)

# COMMAND ----------

# Load events data
events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{CLEAN_TABLE}")
print(events.count())

events = events.dropDuplicates(['SOURCEURL'])
print(events.count())

# COMMAND ----------

# convert data to pandas
news = events.toPandas()

# Minor cleaning
errors = news.loc[:, 'empty_return'] + news.loc[:, 'error_text'] + news.loc[:, 'error_title']
news = news.loc[(errors == 0),:]
news = news.sort_values('DATEADDED').reset_index(drop=True)
news.loc[:, 'rough_token_count'] = news.loc[:, 'text_clean'].map(lambda x: len(x.split()))

sample_news = news.sample(20, random_state=21)

# COMMAND ----------

news.loc[:, 'rough_token_count'].agg(['mean', 'max', 'min'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Summarisation 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.1 SHORT SUMMARISATION

# COMMAND ----------

#note of all prompts used
#prompt = ["tl;dr "] #will give you a very short summary
#prompt = ["tl;dr news articles in 90-100 words"]
#prompt = ["Summarize news articles in two-three complete lines"]
#prompt = ["Summarize text in two-three lines"] 
#prompt = ["Summarize news articles in two-three complete lines. Avoid new lines and lists"] #last part did not work
###prompt = ["Provide a summary of the article in less than 90-100 words"]
#prompts_title = ["tl;dr",
                 #"Expand on the following news article headline into a detailed summary, making sure to incorporate all of the detail. ",
                 #"Act as a news analyst and expand on the following news article headline into a detailed summary, making sure to incorporate all of the detail."]

# COMMAND ----------

#prompt = "Provide a summary of the following text that captures its main idea."
prompts_text = ["tl;dr",
           "Provide a summary of the following text that captures its main idea",
           "Act as a news analyst and provide a summary of the following text that captures its main idea",
           "Act as a news analyst and provide an objective, paragraph-long summary capturing the main idea of the following article for news commentary",
           '''Act as a news analyst and provide an objective, paragraph-long summary capturing the main idea of the following article for news commentary. Your answer should have "The article discusses" as your first starting words''']


# COMMAND ----------

# instantiate
s = Summariser(deployment_name, sample_news)

# COMMAND ----------

# summarize
for prompt in prompts_text:
    s.generate_summaries('text_clean', prompt=prompt, max_tokens=100, temp=1, top_p=1, freq_p=0, presence_p=0, best_of=3, stop=None)

# COMMAND ----------

display(s.args_df)

# COMMAND ----------

# get data for saving
arg_ids = s.args_df.loc[:,"arg_id"].tolist()
summary_df = pd.DataFrame()
for arg_id in arg_ids:
    df, _ = s._get_final_table('text_clean', arg_id)
    summary_df = pd.concat([summary_df, df])

# COMMAND ----------

display(summary_df)

# COMMAND ----------

# convert to pyspark
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spdf = spark.createDataFrame(df)
spar = spark.createDataFrame(ar)

# COMMAND ----------

# save file

OUTPUT_TABLE_NAME = 'gdelt_news_su_short_gold'
ARG_TABLE = 'gdelt_news_su_arg_ids'

spdf.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, SUMMARY_TABLE))
spar.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, ARG_TABLE))

# COMMAND ----------

print(spdf.count())
print(spdf.filter((spdf.openai_summary!='EM') & (spdf.openai_summary!='ER') & (spdf.openai_summary!='NA')).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.2 Param trials

# COMMAND ----------

test_urls = ['https://pmnch.who.int/news-and-events/news/item/11-09-2023-to-improve-my-wellbeing', 'http://www.bjreview.com.cn/Lifestyle/202309/t20230915_800342386.html', 'https://www.wired.com/story/a-global-surge-in-cholera-outbreaks-may-be-fueled-by-climate-change/']
sample_df = news.loc[(news.loc[:,'SOURCEURL'].isin(test_urls)), :].reset_index(drop=True)
display(sample_df)

# COMMAND ----------

# set up prompt
prompt = "Act as a news analyst and provide an objective, paragraph-long summary capturing the main idea of the following article for news commentary."

# COMMAND ----------

### Compare temperature
temps = [0.2, 0.5, 0.8]
s_temp = Summariser(deployment_name, sample_df)
result_df = pd.DataFrame()

for temp in temps:
    s_temp.generate_summaries('text_clean', prompt=prompt, max_tokens=100, temp=temp, top_p=1, freq_p=0, presence_p=0, best_of=1, stop=None)
    arg_id = s_temp.args_df.iloc[0,0]
    df, _ = s_temp._get_final_table('text_clean', arg_id)
    df.loc[:, 'temperature'] = temp
    result_df = pd.concat([result_df, df])
    
display(result_df)

# COMMAND ----------

### Compare presence penalty
presence_ps = [-1.5, 0, 1.5]
s_pre = Summariser(deployment_name, sample_df)
result_df = pd.DataFrame()

for presence_p in presence_ps:
    s_pre.generate_summaries('text_clean', prompt=prompt, max_tokens=100, temp=1, top_p=1, freq_p=0, presence_p=presence_p, best_of=1, stop=None)
    arg_id = s_pre.args_df.iloc[0,0]
    df, _ = s_pre._get_final_table('text_clean', arg_id)
    df.loc[:, 'presence penalty'] = presence_p
    result_df = pd.concat([result_df, df])
    
display(result_df)

# COMMAND ----------

### Compare best of
best_ofs = [1, 2, 3]
s_bef = Summariser(deployment_name, sample_df)
result_df = pd.DataFrame()

for best_of in best_ofs:
    s_bef.generate_summaries('text_clean', prompt=prompt, max_tokens=100, temp=1, top_p=1, freq_p=0, presence_p=0, best_of=best_of, stop=None)
    arg_id = s_bef.args_df.iloc[0,0]
    df, _ = s_bef._get_final_table('text_clean', arg_id)
    df.loc[:, 'best of'] = best_of
    result_df = pd.concat([result_df, df])
    
display(result_df)

# COMMAND ----------

### default setting
s = Summariser(deployment_name, sample_df)
s.generate_summaries('text_clean', prompt=prompt, max_tokens=100, temp=1, top_p=1, freq_p=0, presence_p=0, best_of=1, stop=None)
arg_id = s.args_df.iloc[0,0]
df, _ = s._get_final_table('text_clean', arg_id)
display(df)
