import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
train = pd.read_csv('train.txt',sep=';',names=['Line','Emotion'])
test = pd.read_csv('test.txt',sep=';',names=['Line','Emotion'])
X = train['Line']
x2 = X.head(20)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

sentiment = list()
for sentence in x2:
    sent_vader = list(SentimentIntensityAnalyzer().polarity_scores(sentence).values())
    sentiment.append(sent_vader[3])

sentiment_sentences = pd.DataFrame({'sentence':x2,'sentiment':sentiment})
print(sentiment_sentences)