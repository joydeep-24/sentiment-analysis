import praw
from praw import reddit
user_agent = "Reddit Sentiment analysis by u/jdmandal24"
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_SECRET_KEY",
    user_agent=user_agent
)
import pandas as pd
import numpy as np
import re #RegEx : Regular Expression

headlines = set()
for submission in reddit.subreddit("hentai").hot(limit=None):
    headlines.add(submission.title)
# print(len(headlines))

headlines_df = pd.DataFrame(headlines)
headlines_df.columns = ["Titles"]
headlines_df.drop_duplicates(subset="Titles", inplace=True)

def clean_text(text):
    text = text.lower() # 1. Convert to lowercase
    text = re.sub(r'https?:\/\/\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text) # 2. Remove punctuation and numbers
    return text
headlines_df["Titles"] = headlines_df["Titles"].apply(clean_text)

def remove_emoji(string):
 emoji_pattern = re.compile("["
 u"\U0001F600-\U0001F64F" 
 u"\U0001F300-\U0001F5FF" 
 u"\U0001F680-\U0001F6FF" 
 u"\U00002500-\U00002BEF" 
 u"\U00002702-\U000027B0"
 u"\U00002702-\U000027B0"
 u"\U000024C2-\U0001F251"
 u"\U0001f926-\U0001f937"
 u"\U00010000-\U0010ffff"
 u"\u2640-\u2642"
 u"\u2600-\u2B55"
 u"\u200d"
 u"\u23cf"
 u"\u23e9"
 u"\u231a"
 u"\ufe0f" 
 u"\u3030"
 "]+", flags=re.UNICODE)
 return emoji_pattern.sub(r'', string)
headlines_df["Titles"] = headlines_df["Titles"].apply(remove_emoji)

from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
   return TextBlob(text).sentiment.polarity

headlines_df["Subjectivity"] = headlines_df["Titles"].apply(getSubjectivity)
headlines_df["Polarity"] = headlines_df["Titles"].apply(getPolarity)

def getInsight(score):
   if score < 0:
      return "Negative"
   elif score == 0:
      return "Neutral"
   else:
      return "Positive"
headlines_df["Insight"] = headlines_df["Polarity"].apply(getInsight)

import seaborn as sns
import warnings
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

plt.title("Sentiment Scores")
plt.xlabel("Sentiment")
plt.ylabel("Scores")
plt.rcParams["figure.figsize"] = (10, 8)
headlines_df["Insight"].value_counts().plot(kind="bar", color="#07F5F5")
# plt.show()
stopwords = STOPWORDS
text = ' '.join( [twts for twts in headlines_df['Titles']] )
wordcloud = WordCloud(width=1000, height=600, max_words=100, stopwords=stopwords, background_color='black').generate(text)
plt.figure(figsize=(20,10), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
