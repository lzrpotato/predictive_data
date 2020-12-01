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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
#from scikits.learn.svm.sparse import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
import ast
from anytree import Node, RenderTree, PreOrderIter
import os
from typing import List, Union, Mapping
import numpy as np
import pickle
from tqdm import tqdm


class MyNode():
    id: int
    sid: int
    t: float
    def __init__(self, id, sid, t):
        if id == 'ROOT':
            self.id = 0
        else:
            self.id = int(id)
        if sid == 'ROOT':
            self.sid = 0
        else:
            self.sid = int(sid)
        self.t = float(t)
        
    def __repr__(self):
        return str(self.sid) + '_' + str(self.t)


def build_tree(file):

    root = None
    nodemap = {}
    #'rumor_detection_acl2017/twitter15/tree/' + 
    with open(file, mode='r') as f:
        for line in f:
            splitted = line.split('->')
            p = ast.literal_eval(splitted[0])
            c = ast.literal_eval(splitted[1])
            np = Node(MyNode(*p))
            
            if root is None and np.name.id == 0:
                root = Node(MyNode(*c))
                #print(root.name.id)
                nodemap[root.name.id] = root
                continue
            
            if np.name.id not in nodemap:
                nodemap[np.name.id] = np
            myp = nodemap[np.name.id]
            
            nc = Node(MyNode(*c), parent = myp)
            nodemap[nc.name.id] = nc
            
        return root

def read_tree(path):
    tree_map = {}
    
    for f in os.listdir(path):
        index = f.split('.')[0]
        tree_map[int(index)] = build_tree(os.path.join(path,f))
        
    return tree_map

def encode_tree(tree_map: Mapping[str, Node], max_length=500, padding=False):
    encoded_trees = {}
    for index in sorted(tree_map.keys()):
        root = tree_map[index]
        root_t = root.name.t
        encoding = []

        for i, node in enumerate(PreOrderIter(root)):
            if max_length != -1 and i >= max_length:
                break
            
            if node.name.t-root_t < 0:
               continue
            encoding.append(node.name.t-root_t)

        en_log = np.log10(np.array(encoding)+1)

        encoding = en_log
        if padding:
            len_e = len(encoding)
            if max_length - len_e > 0:
               encoding = np.pad(encoding, (0, max_length-len_e))
        
        encoded_trees[index] = encoding
    
    #print(encoded_tree.shape)
    return encoded_trees
            
def combine_data(content, trees, labels):
    data = []
    """
    #modify/transform text
    cont = dict((k, v.lower()) for k, v in content.items())
        #break into set of words
    cont = word_tokenize(cont)
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(cont):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    
    Tfidf_vect = TfidfVectorizer()
    Tfidf_vect.fit(Final_words)
    cont = Tfidf_vect.transform(Final_words)
    
    print(cont)
    
    #print(Final_words)
    """
    ########
    
    for id, text in content.items():
        label = labels[id]
        tree = trees[id]
        
        data.append([text, tree, label])
        
    data = np.array(data)
   # print(data)
    print(len(content.values()))
    print(tree.shape)
    print(data.shape)
    
    return data[:, 0:2], data[:, 2]

def read_content(path):
    pairs = {}
    with open(path, mode='r', encoding = 'utf-8') as f:

        for line in f:
            
            id, text = line.split('\t')
            
            
            #modify/transform text
            text = text.lower()
            #break into set of words
            text = word_tokenize(text)
            tag_map = defaultdict(lambda : wn.NOUN)
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV
            
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(text):
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words.append(word_Final)
            
            Tfidf_vect = TfidfVectorizer()
            Tfidf_vect.fit(Final_words)
            text = Tfidf_vect.transform(Final_words)
            #print(text)
            #print(Final_words)
            
            ########
            if id not in pairs.keys():
                
                
                
                pairs[int(id)] = text
                #print(text)
            else:
                print('error')
                
    #print(pairs.values())
    return pairs

def read_label(path):
    pairs = {}
    with open(path, mode='r') as f:
        for line in f:
            label, id = line.split(':')
            if id not in pairs.keys():
                pairs[int(id)] = label
            else:
                print('error')
    return pairs

if __name__ == '__main__': 
    
    #tree structure
    
    r_label = read_label('rumor_detection_acl2017/twitter15/label.txt')
    r_text = read_content('rumor_detection_acl2017/twitter15/source_tweets.txt')
    tree_map = read_tree('rumor_detection_acl2017/twitter15/tree/')
    #print(tree_map.shape)
    encoded = encode_tree(tree_map, 500, padding = True)
    #print(encoded.shape)
    
    data = combine_data(r_text, encoded, r_label)
    #print(data[0][0])
    d = []
    for txt, tree in data[0]:
        d.append(np.concatenate((txt, tree), axis = None))
    
    d = np.array(d)
    print(d)
    #print(data[0][0][1])
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(d,data[1],test_size=0.2)
    #print(Train_X)
    #print(Train_X.shape)
    #print(Train_Y.shape)
    #print(Train_Y)

    #d0 = [row[0] for row in data[0]]
    #print(Train_X)
    #Tfidf_vect = TfidfVectorizer()
    #Tfidf_vect.fit(data[0])
    #print(d0[0])
    #Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    #Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    
    
    #print(Train_X.shape)
    #print(Train_Y.shape)
    #print(Train_X[0])
    #print(d.shape)
    
    #svm
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X,Train_Y)
    predictions_SVM = SVM.predict(Test_X)
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
    print(classification_report(Test_Y, predictions_SVM))


"""
    #without tree structure twitter data
    
    news = pd.read_csv('twitter16.csv', encoding='latin-1')
    
    
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
    #print(news)
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(news['text_final'],news['type'],test_size=0.2)
    #print(Train_X)
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(news['text_final'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    #print(Train_X_Tfidf)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    #print(Train_X_Tfidf)
    
    #svm
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
    print(classification_report(Test_Y, predictions_SVM))
    
    #naive bayes
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf,Train_Y)
    predictions_NB = Naive.predict(Test_X_Tfidf)
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
    
    print(classification_report(Test_Y, predictions_NB))
    
    #cm = confusion_matrix(Test_Y, predictions_SVM)
    #print(cm)
    
    
    
    model = SVC()
    ovr = OneVsRestClassifier(model)
    ovr.fit(Train_X_Tfidf, Train_Y)
    predictions_ovr = ovr.predict(Test_X_Tfidf)
    print("OVR Accuracy Score -> ",accuracy_score(predictions_ovr, Test_Y)*100)
    
    print(classification_report(Test_Y, predictions_ovr))
    
    model = SVC()
    ovo = OneVsOneClassifier(model)
    ovo.fit(Train_X_Tfidf, Train_Y)
    predictions_ovo = ovo.predict(Test_X_Tfidf)
    print("OVO Accuracy Score -> ",accuracy_score(predictions_ovo, Test_Y)*100)
    
    print(classification_report(Test_Y, predictions_ovo))

"""



