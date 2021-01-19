import sys
sys.path.append('.')

from lib.utils.twitter_data import TwitterData
from lib.utils.status_sqlite import Status
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--tree',type=str, default='none',help='propagation tree setting, you can choose either "none" or "tree"')
parser.add_argument('--split_type',type=str,default='15_tv', help='options "15_tv","16_tv","15_tvt","16_tvt"')
parser.add_argument('--fold_index',type=str, default='0',help='fold index, choose from 0-4')
parser.add_argument('--s',action='store_true',dest='subclass')
parser.add_argument('--method',type=str, default='SVM',help='method "SVM","RF"')
parser.add_argument('--text', type=str, default='token', help='text format, options "token","raw"')
args = parser.parse_args()

tree = args.tree
fold = args.fold_index
split_type = args.split_type
method = args.method
n_estimators = 10
subclass = args.subclass
text = args.text

td = TwitterData(tree=tree,max_tree_length=100,split_type=split_type,cv=True,datatype='numpy',subclass=subclass,textformat=text)
td.setup()


ss = Status()
dataset = '_'.join(['subclass' if subclass else 'all',tree,split_type,text])
print(dataset)
p = {'CurEpoch':-1,'acc':-1,'c1':-1,'c2':-1,'c3':-1,'c4':-1,'method':method,'dataset':dataset,'fold':-1,'ok':True}
for i in td.kfold_gen():
    print(f'fold {i}')
    if method == 'SVM':
        #if subclass:
        clf = SVC()
        #else:
        #    clf = OneVsRestClassifier(BaggingClassifier(SVC(),max_samples=1.0 / n_estimators, n_estimators=n_estimators,n_jobs=1),n_jobs=1)
    else:
        clf = RandomForestClassifier(n_jobs=1)
    x_train, t_train, y_train = td.train_data
    vectorizer = TfidfVectorizer(lowercase=True,max_features=5000)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test, t_test, y_test = td.test_data
    x_test_tfidf = vectorizer.transform(x_test)
    
    yhat_test = None
    if t_train is not None:
        print('t_train is not none')
        x_train_f = np.concatenate((x_train_tfidf.toarray(),t_train),axis=1)
        x_test_f = np.concatenate((x_test_tfidf.toarray(),t_test),axis=1)
    else:
        print('t_train is none')
        x_train_f = x_train_tfidf.toarray()
        x_test_f = x_test_tfidf.toarray()
    
    #pca = PCA(0.95)
    #x_train_pca = pca.fit_transform(x_train_f)
    #x_test_pca = pca.transform(x_test_f)

    clf.fit(x_train_f,y_train)
    yhat_test = clf.predict(x_test_f)
    results = score(y_test,yhat_test)
    f1 = results[2]
    acc = accuracy_score(y_test,yhat_test)
    if subclass:
        p.update({'acc': acc, 'c2':f1[0], 'c3':f1[1]}) 
    else:
        p.update({'acc': acc, 'c1':f1[1], 'c2':f1[0], 'c3':f1[2], 'c4': f1[3]})
    p['fold'] = i
    print(p)

    ss.save_status(p)

results = ss.read_status(p)
print(results)