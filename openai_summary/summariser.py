# author: forogh akbari, email: akbariforogh66@gmail.com
# update: 07/07/2023
# edit: sun-joo lee, sun-joo.lee@undp.org

import pandas as pd
import math
import re
import uuid
import openai
from dotenv import load_dotenv
load_dotenv()


class Summariser:
    def __init__(self, deployment_name, article_df):
        """
        Initialize the Summariser class.
        Args:
            deployment_name: string designating openai deployment 
            article_df (pd.DataFrame): DataFrame containing the articles to be summarized.
        """
        self.deployment_name = deployment_name
        self.article_df = article_df
        self.sampled_df = None
        self.args_df = pd.DataFrame()


    def _stop_sentence(self, text):
        """
        Truncate the text at the last full sentence (ending in a period)
        """
        if len(text) > 0:
            # Find the index of the last period
            last_period_index = text.rfind('.')
            # Extract the substring until the last period
            new_text = text[:last_period_index+1].strip()
        else:
            new_text = 'NA'
        return new_text
    
    def _nchars_leq_ntokens_approx(self, maxTokens):
        """
        Returns a number of characters very likely to correspond <= maxTokens
        """
        sqrt_margin = 0.5
        lin_margin = 1.010175047 #= e - 1.001 - sqrt(1 - sqrt_margin) #ensures return 1 when maxTokens=1
        return max( 0, int(maxTokens*math.exp(1) - lin_margin - math.sqrt(max(0,maxTokens - sqrt_margin) ) )) 

    def _truncate_text_to_maxTokens_approx(self, text, maxTokens):
        """
        Returns a truncation of text to make it (likely) fit within a token limit
        So the output string is very likely to have <= maxTokens, no guarantees though.
        """
        char_index = min(len(text), self._nchars_leq_ntokens_approx(maxTokens))
        text = text[:char_index]
        return self._stop_sentence(text)


    def _print(self, text_col):
        """
        Print text and all the variations of summaries
        """
        for i, row in self.article_df.iterrows():
            print('ORIG TEXT', i)
            print(row[text_col])
            print()
            for j, arg_row in self.args_df.iterrows():
                arg_id = arg_row['arg_id']
                print('SUMMARY', arg_id)
                print(row[f'short_summary_{arg_id}'])
                print()
                print()
                print()
                print() 

    
    def _get_final_table(self, text_col, arg_id):
        """
        Return summary and prompt argument tables for saving to database
        """
        # colnames
        sum_col_full = f'short_summary_full_{arg_id}'
        sum_col = f'short_summary_{arg_id}'
        cols = ['SOURCEURL', 'title', text_col, sum_col_full, sum_col]

        # data
        data = self.article_df[cols].copy()
        data = data.rename(columns={sum_col_full: 'openai_summary_full', sum_col: 'openai_summary'})
        data['arg_id'] = arg_id

        # arg table
        arg_line = self.args_df[self.args_df['arg_id']==arg_id].copy()
        return data, arg_line

    
    def generate_summaries(self, text_col, prompt, max_tokens, temp=0.3, top_p=1, freq_p=0, presence_p=0, best_of=1, stop=None): 
        """
        Generate summaries for the provided DataFrame.
        Adds short_summary_full_X and short_summary_X columns to processed_df attribute
        Args:
            text_col (string): Name of the column containing the article texts.
            prompt (string): prompt indicating direction for open ai.
            max_tokens (integer): max tokens for return text (the summary).
            temp (float): Temperature parameter for controlling the randomness of the generated text.
            top_p (float): Top-p parameter for nucleus sampling.
            freq_p (float): Frequency penalty to discourage repetitive phrases in the generated text.
            presence_p (float): Presence penalty to encourage the model to include specified phrases.
            best_of (int): Number of completions to generate and select the best from.
            stop (str): Text string to stop the generation.
        """

        # Create a copy of the processed DataFrame
        processed_df = self.article_df.copy()  
        
        # make an id for this run
        arg_id = str(uuid.uuid4().fields[-1])[:5]
        args = pd.DataFrame({'arg_id': [arg_id], 'prompt': [prompt], 'max_tokens':[max_tokens], 'temp': [temp], 'top_p': [top_p], 'freq_p': [freq_p], 'presence_p': [presence_p], 'best_of': [best_of], 'stop': [str(stop)]})
        self.args_df = pd.concat([self.args_df, args])

        # summary column
        sum_col_full = f'short_summary_full_{arg_id}'
        sum_col = f'short_summary_{arg_id}'
        sum_col_full_lst = []
        sum_col_lst = []

        # as weird text comes up if you do not hard code NA it
        for index, row in processed_df.iterrows():
            article_text = row[text_col]
            if (article_text == 'None') or (article_text is None) or (article_text == ''):
                sum_col_full_lst.append('NA')
                sum_col_lst.append('NA')
                continue
            
            # construct input text combining prompt and the article text
            input_text = prompt + ' ' + article_text                   
            calc_max_tokens = 2049 - max_tokens

            # cut to token limit and last full sentence
            input_text = self._truncate_text_to_maxTokens_approx(input_text, calc_max_tokens)

            try:
                #generate summary
                response = openai.Completion.create(
                    engine=self.deployment_name, 
                    prompt=input_text,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_p=top_p,
                    frequency_penalty=freq_p,
                    presence_penalty=presence_p,
                    best_of=best_of,
                    stop=stop
                )

                # get summary
                summary = response.choices[0].text.strip()
                if len(summary) == 0:
                    sum_col_full_lst.append('EM')
                    sum_col_lst.append('EM')
                else:
                    sum_col_full_lst.append(summary)
                    sum_col_lst.append(self._stop_sentence(summary))

            except Exception as e:
                sum_col_full_lst.append('ER')
                sum_col_lst.append('ER')
                print(f"Error occurred at index {index}: {str(e)}")

        # save as columns    
        processed_df[sum_col_full] = sum_col_full_lst
        processed_df[sum_col] = sum_col_lst
        self.article_df = processed_df


    def sample_article(self, event_root_code_col, date_col, text_col, source_url_col, samp_num=5, random_state=42):
        """
        Sample articles grouped by event_root_code_col, date_col
        Saves dataframe as processed_df attribute
        Args:
            event_root_code_col (str): Name of the column containing event root codes.
            date_col (str): Name of the column containing the dates.
            text_col (str): Name of the column containing the article texts.
            source_url_col (str): Name of the column containing the source URLs.
            samp_num (integer): number of articles to sample from each group
            random_state (integer): random seed
        """
        data = self.article_df.copy()
        
        #set up for sampling df
        sampled_df = pd.DataFrame(columns=[event_root_code_col, date_col, text_col, source_url_col])

        #for each event root code and per month
        for event_root_code in data[event_root_code_col].unique():
            articles = data[data[event_root_code_col] == event_root_code]
            articles_grouped_by_month = articles.groupby(pd.Grouper(key=date_col, freq='M'))

            #loop through for each month
            for month, articles_month in articles_grouped_by_month:
                articles_month = articles_month[articles_month[text_col].notnull()]
                if articles_month.shape[0] >= samp_num:
                    sampled_data = articles_month.sample(n=samp_num, random_state=random_state)
                    sampled_df = pd.concat([sampled_df, sampled_data])
                elif articles_month.shape[0] > 0:
                    sampled_df = pd.concat([sampled_df, articles_month])
                else:
                    # If all articles have a null text, skip this month
                    continue

        sampled_df.reset_index(drop=True, inplace=True)
        self.sample_df = sampled_df
  
    
    def gen_sum_of_sum(self, text_col, event_root_code_col, date_col, url_col, prompt, max_tokens, temp=0.3, top_p=1, freq_p=0, presence_p=0, best_of=1, stop=None):
        """
        Generate summary of summaries

        Args:
            text_col (str): Name of the column containing the article texts.
            event_root_code_col (str): Name of the column containing event root codes.
            date_col (str): Name of the column containing the dates.
            url_col (str): Name of column containing the urls.
            prompt (str): Prompt indicating direction for open ai.
            max_tokens (int): Maximum number of tokens in the generated summaries.
            temp (float): Temperature parameter for controlling the randomness of the generated text.
            top_p (float): Top-p parameter for nucleus sampling.
            freq_p (float): Frequency penalty to discourage repetitive phrases in the generated text.
            presence_p (float): Presence penalty to encourage the model to include specified phrases.
            best_of (int): Number of completions to generate and select the best from.
            stop (str): Text string to stop the generation.

        Returns:
            dataframe containing summary of summaries
            dataframe containing prompt arguments
        """
        # make an id for this run
        arg_id = str(uuid.uuid4().fields[-1])[:5]
        args = pd.DataFrame({'arg_id': [arg_id], 'prompt': [prompt], 'max_tokens':[max_tokens], 'temp': [temp], 'top_p': [top_p], 'freq_p': [freq_p], 'presence_p': [presence_p], 'best_of': [best_of], 'stop': [str(stop)]})

        dates_lst = []
        codes_lst = []
        urls_lst = []
        sum_col_lst = []
        sum_col_full_lst = []

        # groupby month, event code and cycle through
        summary_df = self.article_df.copy()
        summary_df = summary_df[(summary_df[text_col]!='EM') & (summary_df[text_col]!='ER')]
        summary_df = summary_df.groupby([pd.Grouper(key=date_col, freq='M'), event_root_code_col]).agg({text_col: ' '.join, url_col: list})
        for index, row in summary_df.iterrows():
            dates_lst.append(index[0].date())
            codes_lst.append(index[1])
            urls_lst.append(row[url_col])

            sum_article = row[text_col]
            input_text = prompt + ' ' + sum_article
            try:
                response = openai.Completion.create(
                    engine=self.deployment_name,
                    prompt=input_text,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_p=top_p,
                    frequency_penalty=freq_p,
                    presence_penalty=presence_p,
                    best_of=best_of,
                    stop=stop)

                summary = response.choices[0].text.strip()
                
                if len(summary) == 0:
                    sum_col_full_lst.append('EM')
                    sum_col_lst.append('EM')
                else:
                    sum_col_full_lst.append(summary)
                    sum_col_lst.append(self._stop_sentence(summary))

            except Exception as e:
                sum_col_full_lst.append('ER')
                sum_col_lst.append('ER')
                print(f"Error occurred at index {index}: {str(e)}")
        
        sum_df = pd.DataFrame({'Year_Month': dates_lst, 'EventRootCode': codes_lst, 'URLs': urls_lst, 'Summary_Full': sum_col_full_lst, 'Summary_Summary': sum_col_lst})
        sum_df['arg_id'] = arg_id

        return sum_df, args

