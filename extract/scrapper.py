# Base scraper

# Author: Linwood Creekmore
# Email: valinvescap@gmail.com
# Description:  Python script to pull content from a website (works on news stories).

# Licensed under GNU GPLv3; see https://choosealicense.com/licenses/lgpl-3.0/ for details

# Notes
"""
23 Oct 2017: updated to include readability based on PyCon talk: https://github.com/DistrictDataLabs/PyCon2016/blob/master/notebooks/tutorial/Working%20with%20Text%20Corpora.ipynb
18 Jul 2018: added keywords and summary
"""

###################################
# Standard Library imports
###################################

import re
import pytz
import datetime
import platform

###################################
# Third party imports
###################################

import requests
import nltk
nltk.download('punkt')
from newspaper import Article
from bs4 import BeautifulSoup
from readability.readability import Document as Paper
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def textgetter(url):
    """Scrapes web news and returns the content
    Parameters
    ----------
    url : str
        web address to news report
    Returns 
    -------
    
    answer : dict
        Python dictionary with key/value pairs for:
            text (str) - Full text of article
            url (str) - url to article
            title (str) - extracted title of article
            author (str) - name of extracted author(s)
            base (str) - base url of where article was located
            provider (str) - string of the news provider from url
            published_date (str,isoformat) - extracted date of article
            top_image (str) - extracted url of the top image for article
    """

    TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']

    # regex for url check
    s = re.compile('(http://|https://)([A-Za-z0-9_\.-]+)')
    u = re.compile("(http://|https://)(www.)?(.*)(\.[A-Za-z0-9]{1,4})$")
    if s.search(url):
        site = u.search(s.search(url).group())
        if u.search(s.search(url).group()) is not None:
            site = site.group(3)
    else:
        site = None
    
    # collect answer 
    answer = {}
    
    # check that it's a url
    if s.search(url):
        try:
            # make a request to the url: INCREASE TIMEOUT IF IF TOO MANY FAILS
            r = requests.get(url, verify=False, timeout=10)
        except:
            # if the url does not return data, set to empty values
            # answer['author'] = None
            answer['base'] = s.search(url).group()
            answer['provider'] = site
            # answer['published_date'] = None
            answer['text'] = ''
            answer['title'] = ''
            # answer['top_image'] = None
            answer['url'] = url
            answer['keywords'] = []
            answer['summary'] = ''
            answer['scrape'] = 'Request failed'
            yield answer
            
        # if url does not return successfully, set to empty values
        if r.status_code != 200:
            # answer['author'] = None
            answer['base'] = s.search(url).group()
            answer['provider'] = site
            # answer['published_date'] = None
            answer['text'] = ''
            answer['title'] = ''
            # answer['top_image'] = None
            answer['url'] = url
            answer['keywords'] = []
            answer['summary'] = ''
            answer['scrape'] = 'Request not OK'

        # test if length of url content is greater than 500, if so, fill data
        if len(r.content) > 500:
            # set article url
            article = Article(url)
            # test for python version because of html different parameters
            if int(platform.python_version_tuple()[0])==3:
                article.download(input_html=r.content)
            elif int(platform.python_version_tuple()[0])==2:
                article.download(html=r.content)
            # parse the url
            article.parse()
            article.nlp()
            
            # if parse doesn't pull text, fill the rest of the data
            if len(article.text) >= 200:
                # answer['author'] = ", ".join(article.authors)
                answer['base'] = s.search(url).group()
                answer['provider'] = site
                # answer['published_date'] = article.publish_date
                answer['keywords'] = article.keywords
                answer['summary'] = article.summary
                # convert the data to isoformat; exception for naive date
                if isinstance(article.publish_date,datetime.datetime):
                    try:
                        answer['published_date'] = article.publish_date.astimezone(pytz.utc).isoformat()
                    except:
                        answer['published_date'] = article.publish_date.isoformat()
                answer['text'] = article.text
                answer['title'] = article.title
                # answer['top_image'] = article.top_image
                answer['url'] = url
                answer['scrape'] = 'Article package'
              
            # else clause: if parse doesn't pull text (try another library)
            else:
                doc = Paper(r.content)
                data = doc.summary()
                title = doc.title()
                soup = BeautifulSoup(data, 'lxml')
                newstext = " ".join([l.text for l in soup.find_all(TAGS)])

                # as we did above, pull text if it's greater than 200 length
                if len(newstext) > 200:
                    # answer['author'] = None
                    answer['base'] = s.search(url).group()
                    answer['provider'] = site
                    # answer['published_date'] = None
                    answer['text'] = newstext
                    answer['title'] = title
                    # answer['top_image'] = None
                    answer['url'] = url
                    answer['keywords'] = []
                    answer['summary'] = ''
                    answer['scrape'] = 'Soup over 200'
                # if nothing works above, use beautiful soup
                else:
                    newstext = " ".join([
                        l.text
                        for l in soup.find_all(
                            'div', class_='field-item even')
                    ])
                    # answer['author'] = None
                    answer['base'] = s.search(url).group()
                    answer['provider'] = site
                    # answer['published_date'] = None
                    answer['text'] = newstext
                    answer['title'] = title
                    # answer['top_image'] = None
                    answer['url'] = url
                    answer['keywords'] = []
                    answer['summary'] = ''
                    answer['scrape'] = 'Soup under 200'
                    
        # else clause: test if length of url content is greater than 500 (if nothing works, fill with empty values)
        else:
            # answer['author'] = None
            answer['base'] = s.search(url).group()
            answer['provider'] = site
            # answer['published_date'] = None
            answer['text'] = ''
            answer['title'] = ''
            # answer['top_image'] = None
            answer['url'] = url
            answer['keywords'] = []
            answer['summary'] = ''
            answer['scrape'] = 'No text returned'
            yield answer
        yield answer

    # else clause: check that it's a url (invalid url)
    else:
        # answer['author'] = None
        answer['base'] = s.search(url).group()
        answer['provider'] = site
        # answer['published_date'] = None
        answer['text'] = ''
        answer['title'] = ''
        # answer['top_image'] = None
        answer['url'] = url
        answer['keywords'] = []
        answer['summary'] = ''
        answer['scrape'] = 'Not a proper url'
        yield answer