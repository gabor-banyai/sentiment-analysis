"""
In Ensembling.py file the data is read, pre-processed and
ensembled with sentiment analysis and different vectorized models

Authors: P. Bijl, M. Malkoc, A. Abdelghany, G. Banyai
VU Amsterdam, 19 December 2020
"""
import numpy as np
import pandas as pd
import warnings
import string
import nltk
import sys

warnings.filterwarnings('ignore')
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.naive_bayes import MultinomialNB
from Sentiment_Transformers import BlobTransformer, NltkTransformer, DictionaryTransformer

# argument given in cmd for model
# 1=VADER , 2=TextBlob, 3=AFINN, 4= ensembling of sentiment analysis
# 5=logistic regression,6=weighted avg. logistic, blob, svm, nb ,7=max_voting logistic, blob, svm, nb,
# 8=logistic, blob+gradient boosting, svm, 9=logistic, blob+ada boosting, svm
# 10=logistic, blob+bagging, svm
ensemble_model = sys.argv[1]

nltk.download('stopwords')
stop = stopwords.words('english')


# deleting punctuation in text
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


# read in files
depression = pd.read_csv("depression.tsv", sep='\t')
depression['target'] = True  # label
angry = pd.read_csv("angry.tsv", sep='\t')
angry['target'] = False
happy = pd.read_csv("happy.tsv", sep='\t')
happy['target'] = False
hangover = pd.read_csv("hangover.tsv", sep='\t')
hangover['target'] = False
strong_opinion = pd.read_csv("strong_opinion.tsv", sep='\t')
strong_opinion['target'] = False
feel_like_new = pd.read_csv("feel_like_new.tsv", sep='\t')
feel_like_new['target'] = False

# add all data together
data = pd.concat([depression, angry, happy, hangover, strong_opinion, feel_like_new]).reset_index(drop=True)
data = shuffle(data)
data = data.reset_index(drop=True)

# pre-process all the data
data['text'] = data['text'].apply(lambda x: x.lower())  # make all text lowercase
data['text'] = data['text'].apply(punctuation_removal)  # remove all the punctuation
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in
                                                      x.split() if word not in stop]))  # remove stopwords

# Class count
count_class_0, count_class_1 = data.target.value_counts()

# Divide by class
df_class_0 = data[data['target'] == 0]
df_class_1 = data[data['target'] == 1]

# oversampling to make depression dataset as big as the others together
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(df_test_over['text'], df_test_over.target,
                                                    test_size=0.2, random_state=42)


# prediction of the test data given the pipe
def sentiment_prediction(pipe):
    pipe.fit(X_train, y_train)
    prediction = pipe.predict(X_test)
    return prediction


# SE weighted avg. or max voting depending on model
def prediction_se(prediction1, prediction2, prediction3, prediction4=0):
    final_prediction = np.array([])
    for i in range(0, len(X_test)):
        if ensemble_model == "4":
            statement = round(prediction1[i] * 0.33 + prediction2[i] * 0.33 + prediction3[i] * 0.33)
        elif ensemble_model == "6":
            statement = round(prediction1[i] * 0.4 + prediction2[i] * 0.2 + prediction3[i] * 0.2 + prediction4[i] * 0.2)
        elif ensemble_model == "7":
            statement = round(
                prediction1[i] * 0.25 + prediction2[i] * 0.25 + prediction3[i] * 0.25 + prediction4[i] * 0.25)
        else:
            statement = round(prediction1[i] * 0.4 + prediction2[i] * 0.3 + prediction3[i] * 0.3)
        final_prediction = np.append(final_prediction, statement)
    return final_prediction


# depending on model given in cmd creating pipelines and ensembling
if ensemble_model == "1":  # method 1; Sentiment NltkTransformer
    pipe = make_pipeline(NltkTransformer(), LogisticRegression())
    final_pred = sentiment_prediction(pipe)
elif ensemble_model == "2":  # method 2; Sentiment BlobTransformer
    pipe = make_pipeline(BlobTransformer(), LogisticRegression())
    final_pred = sentiment_prediction(pipe)
elif ensemble_model == "3":  # method 3; Sentiment DictionaryTransformer
    pipe = make_pipeline(DictionaryTransformer(), LogisticRegression())
    final_pred = sentiment_prediction(pipe)
elif ensemble_model == "4":  # method 4; max_voting sentiment
    pipe = make_pipeline(NltkTransformer(), LogisticRegression())
    pipe2 = make_pipeline(BlobTransformer(), LogisticRegression())
    pipe3 = make_pipeline(DictionaryTransformer(), LogisticRegression())

    prediction1 = sentiment_prediction(pipe)
    prediction2 = sentiment_prediction(pipe2)
    prediction3 = sentiment_prediction(pipe3)

    final_pred = prediction_se(prediction1, prediction2, prediction3)
elif ensemble_model == "5":  # method 5; logistic regression basic model
    pipe = make_pipeline(CountVectorizer(), TfidfTransformer(), LogisticRegression(C=5, max_iter=1000))
    final_pred = sentiment_prediction(pipe)
elif ensemble_model == "6" or ensemble_model == "7":
    # method 6; weighted avg. with logistic regression, blob sentiment, svm and naive bayes
    # method 7; max_voting with logistic regression, blob sentiment, svm and naive bayes
    pipe = make_pipeline(CountVectorizer(), TfidfTransformer(), LogisticRegression(C=5, max_iter=1000))
    pipe2 = make_pipeline(BlobTransformer(), LogisticRegression())
    pipe3 = make_pipeline(CountVectorizer(), TfidfTransformer(), SGDClassifier())
    pipe4 = make_pipeline(TfidfVectorizer(), MultinomialNB())

    prediction1 = sentiment_prediction(pipe)
    prediction2 = sentiment_prediction(pipe2)
    prediction3 = sentiment_prediction(pipe3)
    prediction4 = sentiment_prediction(pipe4)

    final_pred = prediction_se(prediction1, prediction2, prediction3, prediction4)
else:
    pipe = make_pipeline(CountVectorizer(), TfidfTransformer(), LogisticRegression(C=5, max_iter=1000))
    if ensemble_model == "8":  # method 8; logistic regression, blob+gradient boosting, svm
        pipe2 = make_pipeline(BlobTransformer(), GradientBoostingRegressor())
    elif ensemble_model == "9":  # method 9; logistic regression, blob+ada boosting, svm
        pipe2 = make_pipeline(BlobTransformer(), AdaBoostRegressor())
    elif ensemble_model == "10":  # method 10; logistic regression, blob+bagging, svm
        pipe2 = make_pipeline(BlobTransformer(), BaggingRegressor(tree.DecisionTreeRegressor(random_state=1)))
    pipe3 = make_pipeline(CountVectorizer(), TfidfTransformer(), SGDClassifier())

    prediction1 = sentiment_prediction(pipe)
    prediction2 = sentiment_prediction(pipe2)
    prediction3 = sentiment_prediction(pipe3)

    final_pred = prediction_se(prediction1, prediction2, prediction3)

print("accuracy: {}%".format(round(accuracy_score(y_test, final_pred) * 100, 2)))