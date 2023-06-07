import sys
import random
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import json
import logging
from sklearn.feature_extraction.text 
import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import nltk
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import make_scorer

df = pd.read_csv("train.csv",names = ['pol','id','date','q','user','tw'], sep=",")
df = df.drop(['id', 'date','q','user'], axis=1)

frames = [df[0:500000], df[900000:1400000]]

df = pd.concat(frames, ignore_index=True)
regex = re.compile("[А-Яа-яёA-Za-z]+")
mystopwords = stopwords.words('english')

def remove_a(text):
	text = re.sub(r'[@]\w*',"", text).strip()
	return text

def remove_url(text):
	text = re.sub(r"http\S+", "", text).strip()
	text = re.sub(r"https\S+", "", text).strip()
	return text

def words_only(text, regex=regex):
	return " ".join(regex.findall(text))
	
def remove_stopwords(text, mystopwords = stopwords.words('english')):
	try:
		return " ".join([token for token in text.split() if (token not in mystopwords) and (token.count(token[0]) != len(token))])
	except:
		return ""

def stem_text(text):
	stemmer = nltk.PorterStemmer()
	stemmed_words = list(map(stemmer.stem, text.split(" ")))
	return " ".join(stemmed_words)
	
def emoji(text):
	happy = ["<3"]
	sad = []
	
	eyes = ["8",":","=",";"]
	nose = ["'","`","-",r"\\"]
	for e in eyes:
		for n in nose:
			for s in ["\)", "d", "]", "}","p", ")"]:
				happy.append(e+n+s)
				happy.append(e+s)
			for s in ["\(", "\[", "{", "\|", "\/", "(", "[" r"\\"]:
				sad.append(e+n+s)
				sad.append(e+s)

	happy = list(set(happy))
	sad = list(set(sad))

	t = []
	for word in text.split():
		if word in happy:
			t.append("happy")
		elif word in sad:
			t.append("sad")
		else:
			t.append(word)
	text = " ".join(t)
	return text

positive_word_library = list(set(open('positive-words.txt', encoding = "ISO-8859-1").read().split()))
negative_word_library = list(set(open('negative-words.txt', encoding = "ISO-8859-1").read().split()))

def pos_neg(text):
	t = []
	for word in text.split():
		if word in positive_word_library:
			t.append('positive ' + word)
		elif word in negative_word_library:
			t.append('negative ' + word)
		else:
			t.append(word)
		text = " ".join(t)
		return text

def preprocess(df):
	df.tw = df.tw.apply(remove_a)
	df.tw = df.tw.str.lower()
	df.tw = df.tw.apply(remove_url)
	df.tw = df.tw.apply(stem_text)
	df.tw = df.tw.apply(remove_stopwords)
	df.tw = df.tw.apply(words_only)
	return df

df=preprocess(df)

df.head(5)

train_data, test_data = train_test_split (df, test_size=0.25, random_state=42)

def evaluate_prediction(predictions, target):
	print('accuracy %s' % accuracy_score(target, predictions))
	print('precision %s' % precision_recall_fscore_support(target, predictions)[0])
	print('recall %s' % precision_recall_fscore_support(target, predictions)[1])
	print('fscore %s' % precision_recall_fscore_support(target, predictions)[2])

def predict(vectorizer, classifier, data):
	data_features = vectorizer.transform(data['tw'])
	predictions = classifier.predict(data_features)
	target = data['pol']
	evaluate_prediction(predictions, target)

count_vectorizer = CountVectorizer(
	analyzer="word", tokenizer=nltk.word_tokenize,
	lowercase=True,
	preprocessor=None, stop_words='english', max_features = 44946)

train_data_features = count_vectorizer.fit_transform(train_data['tw'])
train_data_features

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


df_pol = []

for i in range (len(df['tw'])):
	if (analyser.polarity_scores(df['tw'][i])['pos'] >= analyser.polarity_scores(df['tw'][i])['neg']):
		df_pol.append(4)
	else:
		df_pol.append(0)
	correct_answ = 0
	for j in range (len(df['pol'])):
		if (df['pol'][j] == df_pol[j]):
			correct_answ += 1
print (correct_answ/(len(df['pol'])))

def lex(df):
	df.tw = df.tw.apply(remove_a)
	df.tw = df.tw.str.lower()
	df.tw = df.tw.apply(remove_url)
	df.tw = df.tw.apply(remove_stopwords)
	df.tw = df.tw.apply(stem_text)
	df.tw = df.tw.apply(emoji)
	df.tw = df.tw.apply(pos_neg)
	df.tw = df.tw.apply(words_only)
	return df

train_data, test_data = train_test_split (df, test_size=0.25, random_state=42)

count_vectorizer = CountVectorizer(
	analyzer="word", tokenizer=nltk.word_tokenize,
	lowercase=True,
	preprocessor=None, stop_words='english', max_features = 44946)

train_data_features = count_vectorizer.fit_transform(train_data['tw'])
%%time

mnb= MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
mnb = mnb.fit (train_data_features, train_data['pol'])
%%time

predict(count_vectorizer, mnb, test_data)
%%time

logreg = LogisticRegression(C=10, penalty='l2')
logreg = logreg.fit(train_data_features, train_data['pol'])
%%time

predict(count_vectorizer, logreg, test_data)
%%time



svm = LinearSVC(C=100, penalty='l2', dual=False)
svm = svm.fit(train_data_features, train_data['pol'])
%%time

predict(count_vectorizer, svm, test_data)

count_vectorizer = CountVectorizer(
	analyzer="word", tokenizer=nltk.word_tokenize,
	lowercase=True, ngram_range=(1, 2),
	preprocessor=None, stop_words='english', max_features = 44946)
train_data_features = count_vectorizer.fit_transform(train_data['tw'])

mnb= MultinomialNB(alpha =1.0)
mnb = mnb.fit (train_data_features, train_data['pol'])

predict(count_vectorizer, mnb, test_data)

logreg = LogisticRegression(C=10, penalty='l2')
logreg = logreg.fit(train_data_features, train_data['pol'])

predict(count_vectorizer, logreg, test_data)

count_vectorizer = CountVectorizer(
	analyzer="word", tokenizer=nltk.word_tokenize,
	lowercase=True, ngram_range=(2, 2),
	preprocessor=None, stop_words='english', max_features = 44946)
train_data_features = count_vectorizer.fit_transform(train_data['tw'])

mnb= MultinomialNB(alpha =1.0)
mnb = mnb.fit (train_data_features, train_data['pol'])

predict(count_vectorizer, mnb, test_data)

logreg = LogisticRegression(C=10, penalty='l2')
logreg = logreg.fit(train_data_features, train_data['pol'])

predict(count_vectorizer, logreg, test_data)

tfv = TfidfVectorizer(
	strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
	ngram_range=(1, 1),
	stop_words = 'english', max_features = 44946)

train_data_features_tf = tfv.fit_transform(train_data['tw'])

logreg = LogisticRegression(C=10, penalty='l2')
logreg = logreg.fit(train_data_features_tf, train_data['pol'])
%%time



predict(tfv, logreg, test_data)

mnb= MultinomialNB(alpha =1.0)
mnb = mnb.fit (train_data_features_tf, train_data['pol'])%%time

predict(tfv, mnb, test_data)

tfv = TfidfVectorizer(
	strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
	ngram_range=(1, 2),
	stop_words = 'english', max_features = 44946)

train_data_features_tf = tfv.fit_transform(train_data['tw'])

mnb= MultinomialNB(alpha =1.0)
mnb = mnb.fit (train_data_features_tf, train_data['pol'])

predict(tfv, mnb, test_data)

logreg = LogisticRegression(C=10, penalty='l2')
logreg = logreg.fit(train_data_features_tf, train_data['pol'])
%%time

predict(tfv, logreg, test_data)

tfv = TfidfVectorizer(
	strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
	ngram_range=(2, 2),
	stop_words = 'english', max_features = 44946)

train_data_features_tf = tfv.fit_transform(train_data['tw'])

mnb= MultinomialNB(alpha =1.0)
mnb = mnb.fit (train_data_features_tf, train_data['pol'])

predict(tfv, mnb, test_data)

logreg = LogisticRegression(C=10, penalty='l2')
logreg = logreg.fit(train_data_features_tf, train_data['pol'])
%%time

predict(tfv, logreg, test_data)
