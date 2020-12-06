# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:30:01 2020

@author: Penghui
"""

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

news = pd.read_csv('twitter15.csv', encoding='latin-1')


#Data preprocessing

#remove bland rows
news['content'].dropna(inplace=True)
# change all text to lower
news['content'] = [entry.lower() for entry in news['content']]
#break into set of words
news['content'] = [word_tokenize(entry) for entry in news['content']]

#wordnet lemmenting
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(news['content']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    news.loc[index,'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(news['text_final'],news['type'],test_size=0.5)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(news['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#svm
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)



Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)





