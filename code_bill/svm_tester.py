import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.model_selection import (GridSearchCV, ShuffleSplit,RandomizedSearchCV,
                                     cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from lib.transfer_learn.param import Param, ParamGenerator
from lib.utils import Status
import warnings
warnings.filterwarnings("ignore")

class ModelClf():
    def __init__(self, pca, method):
        if pca == 0:
            self.clf = make_pipeline(StandardScaler(), method)
        else:
            self.clf = make_pipeline(StandardScaler(), PCA(n_components=pca), method)

    def train(self, x, y):
        self.clf.fit(x, y)
        return self.clf.score(x,y)

    def test(self,x,y):
        score = self.clf.score(x,y)
        return score

class factory():
    def __init__(self):
        dataset = None
        self.pg = ParamGenerator()
        p = next(self.pg.gen())
        self.status = Status(p.exp)
        
    def cv_model_select(self):
        #
        # self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        for p in self.pg.gen():
            if not self.status.read_key(p,'ok'):
                continue

            with open(f'./features/feamap_{p.for_name}.npz','rb') as f:
                dataset = np.load(f)
            self.x, self.y = dataset[:,0:-1], dataset[:,-1]
            bestscore = 0
            param = None
            for n in range(0,100,10):
                if n == 0:
                    self.clf = make_pipeline(StandardScaler(),RandomForestClassifier())
                else:
                    self.clf = make_pipeline(StandardScaler(),PCA(n),RandomForestClassifier())
                cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
                score = cross_val_score(self.clf, self.x,self.y, cv=cv, n_jobs=-1)
                #print('RF pca ', n,' score ',np.average(score))
                if bestscore < np.average(score):
                    bestscore = np.average(score)
                    param = [n, RandomForestClassifier()]

            x_train, x_test, y_train, y_test = train_test_split(self.x,
                self.y, test_size=0.4, random_state=0)
            
            print('best model', param, f' {p.features} bestscore {bestscore}')

            bestscore = 0
            param = None
            for n in range(0,100,10):
                if n == 0:
                    self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                else:
                    self.clf = make_pipeline(StandardScaler(),PCA(n), SVC(gamma='auto'))
                cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
                score = cross_val_score(self.clf, self.x,self.y, cv=cv, n_jobs=-1)
                #print('RF pca ', n,' score ',np.average(score))
                if bestscore < np.average(score):
                    bestscore = np.average(score)
                    param = [n, SVC(gamma='auto')]
            
            x_train, x_test, y_train, y_test = train_test_split(self.x,
                self.y, test_size=0.4, random_state=0)
            
            print('best model', param, f' {p.features} bestscore {bestscore}')

    def train_with_PCA(self, n_components):
        print('start')
        results = []
        best_param = []
        svm_best_param = None
        rf_best_param = None
        cv_avg = []
        
        for p in self.pg.gen():
            svm_best_param = None
            rf_best_param = None
            fold_results = []
            for fold in range(5):
                svm_best_param = None
                rf_best_param = None
                if not os.path.isfile(f'./features/feamap_train_fd={fold}_{p.experiment_name}.npz'):
                    print(f'[skip] {fold} p {p}')
                    continue
                print(f'fold {fold} p {p}')
                with open(f'./features/feamap_train_fd={fold}_{p.experiment_name}.npz','rb') as f:
                    dataset = np.load(f)
                train_x, train_y = dataset[:,0:-1], dataset[:,-1]
                origin_shape = train_x.shape
                with open(f'./features/feamap_test_fd={fold}_{p.experiment_name}.npz','rb') as f:
                    test_dataset = np.load(f)
                test_x, test_y = test_dataset[:,0:-1], test_dataset[:,-1]
            
                print('finished load data')
                pca = PCA(n_components,svd_solver='full')
                train_x = pca.fit_transform(train_x)
                test_x = pca.transform(test_x)

                ss = StandardScaler()
                train_x = ss.fit_transform(train_x)
                test_x = ss.transform(test_x)

                print(f'PCA nc {pca.n_components_} train.shape {origin_shape}')
                #index = np.random.choice(train_x.shape[0],size=train_x.shape[0],replace=False)
                #train_x_r, train_y_r = train_x[index], train_y[index]
                if svm_best_param is None:
                    clf,svm_best_param = self.grid_search_svc(train_x, train_y)
                else:
                    clf = OneVsRestClassifier(SVC(**svm_best_param),n_jobs=-1)
                clf.fit(train_x,train_y)
                svc_train_score = clf.score(train_x,train_y)
                svc_test_score = clf.score(test_x,test_y)
                print(f'{p.experiment_name} SVC train {svc_train_score} test {svc_test_score}',flush=True)

                if rf_best_param is None:
                    clf,rf_best_param = self.random_search_rf(train_x, train_y)
                else:
                    clf = RandomForestClassifier(n_jobs=-1,**rf_best_param)
                clf.fit(train_x,train_y)
                rf_train_score = clf.score(train_x,train_y)
                rf_test_score = clf.score(test_x,test_y)
                print(f'{p.experiment_name} RF train {rf_train_score} test {rf_test_score}')
                
                results.append([p.experiment_name,fold,svc_train_score,svc_test_score,rf_train_score,rf_test_score])
                best_param.append((p.experiment_name,fold,svm_best_param,rf_best_param,pca.n_components_))

                fold_results.append([svc_test_score,rf_test_score])
            avg = np.mean(fold_results,axis=0)
            
            cv_avg.append([p.split_type,f'{p.max_tree_len}_{p.dnn}',*avg])
        results = pd.DataFrame(results,columns=['fea','fold','svm_train','svm_test','rf_train','rf_test'])
        print(results)
        results.to_csv(os.path.join('./results/',f'pca_{n_components}_results.csv'))
        cv_avg = pd.DataFrame(cv_avg,columns=['data','tree_dnn','svm_test','rf_test'])
        cv_avg.to_csv(os.path.join('./results/',f'pca_{n_components}_cv_avg.csv'))
        print(cv_avg)
        print(best_param)
        with open('./results/'+f'pca_{n_components}_param.txt','w') as f:
            print(best_param, file=f)

    def draw_featuremap(self):
        for i,nr in enumerate([0,17,3,4]):
            fig, axes = plt.subplots(1,3,figsize=(10,5))
            for ind, p in enumerate(self.pg.gen()):
                if not self.status.read_key(p,'ok'):
                    continue
                if len(p.features) != 1:
                    continue
                with open(f'./features/feamap_{p.for_name}.npz','rb') as f:
                    dataset = np.load(f)
                self.x, self.y = dataset[:,0:-1], dataset[:,-1]
                
                s = int(np.sqrt(self.x.shape[1]))
                img = self.x[nr,:].copy()
                img.resize(s*s)
                img = img.reshape(s,s)
                axes[ind].imshow(img)
        
            fn = f'feamap_{i}.png'
            path = os.path.join('./figure/',fn)
            plt.savefig(path)
            print(f'save fig {fn}')
            plt.close()

    def find_pca(self):
        for p in self.pg.gen():
            for fold in range(5):
                if not os.path.isfile(f'./features/feamap_train_fd={fold}_{p.experiment_name}.npz'):
                    continue
                print(f'fold {fold} p {p}')
                with open(f'./features/feamap_train_fd={fold}_{p.experiment_name}.npz','rb') as f:
                    dataset = np.load(f)
                train_x, train_y = dataset[:,0:-1], dataset[:,-1]
                
                with open(f'./features/feamap_test_fd={fold}_{p.experiment_name}.npz','rb') as f:
                    test_dataset = np.load(f)
                test_x, test_y = test_dataset[:,0:-1], test_dataset[:,-1]
                
                print('finished load data')
                pca = PCA(0.98,svd_solver='full')
                pca.fit(train_x)

                evr = pca.explained_variance_ratio_
                print(evr.shape, train_x.shape)
            
        
    def random_search_rf(self,x,y):
        print('start random search rf')
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 5)]
        # Number of features to consider at every split
        # max_features = ['auto', 'sqrt','log2']
        max_features = ['auto']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        
        random_search = RandomizedSearchCV(RandomForestClassifier(),random_grid,n_iter=1000,random_state=0,cv=5,n_jobs=-1,verbose=1) # 
        search = random_search.fit(x,y)
        print('best_param ', search.best_params_)
        clf = RandomForestClassifier(n_jobs=-1,**search.best_params_)
        return clf, search.best_params_

    def grid_search_svm(self,x,y):
        print('start grid search svm')
        Cs = [0.001, 0.01, 0.1, 1, 10]
        #Cs = [1]
        gammas = [0.001, 0.01, 0.1, 1, 'auto','scale']
        #gammas = [0.001]
        param_grid = {'estimator__base_estimator__C': Cs, 'estimator__base_estimator__gamma' : gammas}
        n_estimators = 10
        clf = OneVsRestClassifier(BaggingClassifier(SVC(),max_samples=1.0 / n_estimators, n_estimators=n_estimators,n_jobs=-1),n_jobs=-1)
        #param_grid = {'C': Cs, 'gamma' : gammas}
        #clf = SVC()
        grid_search = GridSearchCV(clf,param_grid,cv=5,n_jobs=-1,verbose=1)
        grid_search.fit(x,y)
        print('best_param ',grid_search.best_params_)
        best_params = {}
        for k,v in grid_search.best_params_.items():
            best_params[k.split('__')[2]] = v
        clf = OneVsRestClassifier(BaggingClassifier(SVC(**best_params),max_samples=1.0 / n_estimators, n_estimators=n_estimators,n_jobs=-1),n_jobs=-1)
        #clf = SVC(**grid_search.best_params_)
        return clf, best_params

    def grid_search_svc(self,x,y):
        print('start grid search svm')
        Cs = [0.001, 0.01, 0.1, 1, 10]
        #Cs = [1]
        gammas = [0.001, 0.01, 0.1, 1, 'auto','scale']
        #gammas = [0.001]
        param_grid = {'estimator__C': Cs, 'estimator__gamma' : gammas}
        clf = OneVsRestClassifier(SVC(),n_jobs=-1)
        #param_grid = {'C': Cs, 'gamma' : gammas}
        #clf = SVC()
        grid_search = GridSearchCV(clf,param_grid,cv=5,n_jobs=-1,verbose=1)
        grid_search.fit(x,y)
        print('best_param ',grid_search.best_params_)
        best_params = {}
        for k,v in grid_search.best_params_.items():
            best_params[k.split('__')[1]] = v
        clf = OneVsRestClassifier(SVC(**best_params),n_jobs=-1)
        #clf = SVC(**grid_search.best_params_)
        return clf, best_params

    def draw_results(self):
        fn_results = {
            'ae_results_85_dense':'Autoencoder 85%', 'ae_results_95_dense':'Autoencoder 95%',
            'pca_0.85_results':'PCA 85%', 'pca_0.95_results':'PCA 95%',
        }
        for fn,title in fn_results.items():
            rst = pd.read_csv('./results/'+fn+'.csv', index_col=[0])
            rst = rst.set_index('fea')
            new_df = rst.loc[:,['svm_test','rf_test']]
            fig, ax = plt.subplots(figsize=(8,5))
            x = np.arange(new_df.shape[0])
            width=0.45
            rects1 = ax.bar(x-width/2, new_df['svm_test'], width, label='SVM')
            rects2 = ax.bar(x+width/2, new_df['rf_test'], width, label='RF')
            ax.set_ylim(0.6,1)
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Accuracies by classifier and ensemble with {title}')
            ax.set_xticks(x)
            ax.set_xticklabels(new_df.index)
            ax.legend()
            def autolabel(rects):
                """Attach a text label above each bar in *rects*, displaying its height."""
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{:.3f}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            autolabel(rects1)
            autolabel(rects2)
            fig.tight_layout()
            plt.savefig('./figure/'+f'ml_results_{fn}.png')
            print('save fig ','./figure/'+f'ml_results_{fn}.png')
            plt.close()

if __name__ == '__main__':
    f = factory()
    #f.draw_results()
    #f.train_with_AE('85')
    f.train_with_PCA(0.98)
    #f.draw_feature_from_image()
    #f.find_pca()

