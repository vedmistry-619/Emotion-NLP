import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import subprocess
train = pd.read_csv('train.txt',sep=';',names=['Line','Emotion'])
test = pd.read_csv('test.txt',sep=';',names=['Line','Emotion'])
X = train['Line']
x2 = X.head(20)

subprocess.Popen(['java','-mx4g','-cp','*','edu.stanford.nlp.pipeline.StanfordCoreNLPServer'],
cwd= "C:\stanford-corenlp-full-2018-02-27", shell=True, stdout= subprocess.DEVNULL,
stderr=subprocess.STDOUT)

from pycorenlp import StanfordCoreNLP

sentences = x2
sentiment = list()

for sentence in sentences:
    nlp = StanfordCoreNLP('http://localhost:9000')
    sentiment_stanford = nlp.annotate(sentence, properties={'timeout': '500000','annotators': 'sentiment', 'outputFormat': 'json'})
    sentiment_stanford = sentiment_stanford['sentences'][0]['sentimentValue']
    sentiment.append(sentiment_stanford)

sentiment_sentences = pd.DataFrame({'sentence':sentences,'sentiment':sentiment})
labels = {"0": "very negative", "1": "negative", "2":"neutral", "3":"positive", "4":"very positive"}
sentiment_sentences['sentiment'] = sentiment_sentences.sentiment.apply(lambda x: labels[x])

sentiment_sentences.to_csv("stanfordnlp.csv", sep = ";", encoding="utf-8",quotechar="'",index=False)