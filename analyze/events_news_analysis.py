# Databricks notebook source
import pandas as pd
import warnings
warnings.simplefilter('ignore', FutureWarning)
from clean.event_news_clean import merge_event_news
from param_spec import CAMEO_TABLE

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Prep

# COMMAND ----------

# Fetch the event-news data
data = merge_event_news(spark)

# COMMAND ----------

# Minor cleaning

# Re-process the event code to match the CAMEO coding format
def match_code(str_code):
    if len(str_code) == 1:
        return f'0{str_code}'
    return str_code

data.loc[:,'EventRootCode'] = data.loc[:,'EventRootCode'].map(match_code)

# Merge data
new_dat = pd.merge(data, CAMEO_TABLE, 'inner', left_on='EventRootCode', right_on='code')

# Produce the data with target columns
final_dat = new_dat.loc[:, ['EventRootCode', 'Events', 'GoldsteinScale', 
                            'NumMentions', 'NumSources', 'NumArticles', 
                            'AvgTone', 'title', 'lang', 'DATEADDED_news']]

# Cast selected variables as floats
for var in ['GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone']:
    final_dat.loc[:, var] = final_dat.loc[:, var].astype(float)

# COMMAND ----------

final_dat.info()

# COMMAND ----------

final_dat.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis

# COMMAND ----------

import matplotlib.pylab as plt

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Occurrence of Event Type
# MAGIC The result indicates that "Make Public Statement" is the most frequent root event type, and "Reduce Relations" is the least during the given time range.

# COMMAND ----------

final_dat.groupby('Events')\
    .agg({'Events':'count'})\
    .rename(columns={'Events': 'Occurrence'})\
    .sort_values(by=['Occurrence'], ascending=False)\
    .plot(kind='barh')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Analysis of the Event Impact
# MAGIC In the given time range, the calculation of the average Goldstein Scale in each event group shows that the impact of "Yield" is expected to be the most positive on the stability of Malawi; the impact of "Fight" is expected to be the most negative on the stability of Malawi.

# COMMAND ----------

final_dat.groupby('Events')\
    .agg({'GoldsteinScale':'mean'})\
    .rename(columns={'GoldsteinScale': 'Mean of Impact Score'})\
    .sort_values(by=['Mean of Impact Score'], ascending=False)\
    .plot(kind='barh')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Analysis of Event Mentions
# MAGIC Combining the number of mentions, information sources, and the articles, "Reject" turns out to be the event that shares the highest mean of mentions; "Provide aid" turns out to be the event that shares the lowest mean of mentions.

# COMMAND ----------

df = final_dat.copy()
df.loc[:, 'Total Mentions'] = df.loc[:, 'NumMentions'] + df.loc[:, 'NumSources'] + df.loc[:, 'NumArticles']

df.groupby('Events')\
    .agg({'Total Mentions': 'mean'})\
    .rename(columns={'Total Mentions': 'Mean of Mentions'})\
    .sort_values(by=['Mean of Mentions'], ascending=False)\
    .plot(kind='barh')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Analysis of Document Tone
# MAGIC All the documents collected in the given time range expessed negative tone towards the arising events.  "Express intent to cooperate" is the event that has a document tone closest to "neutral"; "Fight" has the most negative document tone among all the listed root events.

# COMMAND ----------

final_dat.groupby('Events')\
    .agg({'AvgTone':'mean'})\
    .rename(columns={'AvgTone': 'Mean of AvgTone'})\
    .sort_values(by=['Mean of AvgTone'], ascending=False)\
    .plot(kind='barh')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Word Cloud Analysis for Titles
# MAGIC The result of the word cloud that's generated based on the news titles suggests that in the given time range, "chizuma", "school", "chakwera", "cholera" , and "arrest" turn out to be 5 most prominent words. These words are expected to be related to eduction, politics, enforced intervention, and public health.

# COMMAND ----------

import string
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from wordcloud import WordCloud

# COMMAND ----------

def get_wordnet_pos(treebank_tag):
    """
    Translate nltk POS to wordnet tags
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

    
def lemmatize_word(word):
    """
    Lemmatize a word
    """
    lemmatizer = WordNetLemmatizer()
    nltk_pos_tagged = nltk.pos_tag([word])
    wordnet_pos_tagged = map(lambda x: (x[0], get_wordnet_pos(x[1])), nltk_pos_tagged)
    word, pos = list(wordnet_pos_tagged)[0]
    
    return lemmatizer.lemmatize(word, pos)


def tokenize(text):
    """
    Tokenize the text
    """

    tokens = nltk.word_tokenize(text)
    punctuations = string.punctuation
    stop_words = set(stopwords.words("english"))
    stop_words.update({'’', '–', '...', "'s", "'re", "'d", "'ve", "‘", '''”''', '''“'''})

    new_tokens = []
    
    for token in tokens:
        if (token not in punctuations) and (token.lower() not in stop_words)\
            and (re.match(r"\d+", token) == None):
                lem_token = lemmatize_word(token.lower())
                new_tokens.append(lem_token)
    
    return new_tokens


def gene_token_lst(text_list):
    """
    Return a list containing documents of tokens
    
    Inputs:
        text_list: a list of strings
    """
    
    return [tokenize(text) for text in text_list]


def gene_token_freq(token_doc):
    """
    Generate a dictionary where key is the token and the value is the counts
    of its occurrence.
    
    Inputs:
        token_doc: a list of lists of strings
    Outputs:
        token_freq: a dictionary
    """
    token_freq = dict()
    
    for token_lst in token_doc:
        for token in token_lst:
            if not token in token_freq.keys():
                token_freq[token] = 1
            else:
                token_freq[token] += 1
    
    return token_freq


def plot_wordcloud(token_freq):
    """
    Plot a word cloud based on the frequency data
    
    Inputs:
        token_freq: a dictionary where key is the token and the value is the counts
        of its occurence
    Outputs: a word cloud picture
    """
    
    wc = WordCloud(background_color='white',
                   height=1000,
                   width=1000)
    wc.fit_words(token_freq)
    
    return wc.to_image()

# COMMAND ----------

titles = final_dat.loc[:, 'title'].tolist()
title_tokens = gene_token_lst(titles) # Tokenize the titles
title_freq = gene_token_freq(title_tokens) # Count freq for each token
title_freq.pop('malawi') # 'malawi' as a word is not that significant in this setting
title_cloud = plot_wordcloud(title_freq)

# COMMAND ----------

plt.figure(figsize=(10, 10))
plt.imshow(title_cloud)
plt.axis('off')  # To hide axis
plt.show()
