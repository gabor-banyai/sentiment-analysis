"""
Sentiment_Transformer.py executes different sentiment analysis; VADER, AFINN & TextBlob.
Sentiment analysis is a common NLP task, classifying text into pre-defined sentiments

Authors: A. Abdelghany, G. Banyai, P. Bijl, M. Malkoc
VU Amsterdam, 19 December 2020
"""
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn
from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin


af = Afinn()
sid = SentimentIntensityAnalyzer()


# VADER lexicon and rule base sentiment analysis,
# #especially used to analyse sentiments on social media
class NltkTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    print('\n>>>>>>>init() called.\n')

  def fit(self, X, y = None):
    print('\n>>>>>>>fit() called.\n')
    return self

  def transform(self, X, y = None):
    print('\n>>>>>>>transform() called.\n')
    X_ = X.copy() # creating a copy to avoid changes to original dataset
    # X_ is the list of individual messages
    results = [] # The new array with transformed data, results of sentiment go in here
    for i in X_:
    # i stands for every individual message
    # So below you can use your own sentiment analysis
        ss = sid.polarity_scores(i) # Calculate sentiment
        results.append([ss['compound'],ss['neg'],ss['neu'],ss['pos']]) # Append results
    results = np.array(results)
    return results


# TextBlob sentiment analysis, looking at polarity and subjectivity of a word
class BlobTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    print('\n>>>>>>>init() called.\n')

  def fit(self, X, y = None):
    print('\n>>>>>>>fit() called.\n')
    return self

  def transform(self, X, y = None):
    print('\n>>>>>>>transform() called.\n')
    X_ = X.copy() # creating a copy to avoid changes to original dataset
    # X_ is the list of individual messages
    results = [] # The new array with transformed data, results of sentiment go in here
    for i in X_:
    # i stands for every individual message
        ss = sid.polarity_scores(i) # Calculate sentiment
        results.append([ss['compound'],ss['neg'],ss['neu'],ss['pos']]) # Append results
    results = np.array(results)
    return results


# AFINN sentiment analysis, dictionary based approach
class DictionaryTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    print('\n>>>>>>>init() called.\n')

  def fit(self, X, y = None):
    print('\n>>>>>>>fit() called.\n')
    return self

  def transform(self, X, y = None):
    print('\n>>>>>>>transform() called.\n')
    X_ = X.copy() # creating a copy to avoid changes to original dataset
    # X_ is the list of individual messages
    results = [] # The new array with transformed data, results of sentiment go in here
    for i in X_:
    # i stands for every individual message
        ss = af.score(i) # Calculate sentiment
        results.append([1,ss]) # Append results
        #sentiment_category = ['positive' if score > 0
                            #else 'negative' if score < 0
                            #else 'neutral'
                            #for score in results]
    results = np.array(results)
    return results