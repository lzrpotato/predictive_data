import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from lib.transfer_learn.param import Param, ParamGenerator
from lib.utils.Status import Status
import os

class Visual():
    def __init__(self):
        self.status = Status()
        self.pg = ParamGenerator()

    def _draw_basic(self, df, key, filename):
        sns.set_style('whitegrid')
                
        f, axes = plt.subplots(1,1,figsize=(3,5))
        ax1 = sns.lineplot(data=df[key],markers=True,linewidth=1.5,ax=axes,legend='brief')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(filename)

    def draw_transfer(self):
        results = []
        p: Param
        for p in self.pg.gen():

            _, test_result = self.status.read_best_results(p)
            results.append([p.split_type,f'{p.max_tree_len}_{p.dnn}',p.auxiliary,test_result])

        data = pd.DataFrame(results, columns=['dataset','maxlen_dnn','auxiliary','test_acc'])
        data = data.sort_values(['maxlen_dnn'])
        print(data)
        for dataset, dataset_rest in data.groupby('dataset'):
            print(dataset)
            fig, ax = plt.subplots(figsize=(7,6))
            sns.barplot(x='maxlen_dnn',y='test_acc',hue='auxiliary',data=dataset_rest,ax=ax)
            ax.set_ylim(0.5,1)
            fn = f'barcatplot_{dataset}.png'
            path = os.path.join('./figure',fn)
            print(f'save {path}')
            plt.savefig(path)

    def draw_kfold(self):
        results = []
        p: Param
        for p in self.pg.gen():
            _, test_result = self.status.read_kfold(p)
            if test_result is None:
                continue
            results.append([p.split_type,
                            f'{p.max_tree_len}_{p.dnn}',
                            p.auxiliary,
                            test_result['test_acc_epoch'],
                            test_result['precision'],
                            test_result['recall'],
                            test_result['fscore'],
                            ])
        
        data = pd.DataFrame(results, columns=['dataset','maxlen_dnn','auxiliary',
                                'test_acc','precision','recall','fscore'])
        print(data)
        fn = f'kfold_result_exp={p.exp}.csv'
        data.to_csv(os.path.join('./results/',fn))
        print(f"save dataframe to {os.path.join('./results/',fn)}")

    def draw_svm_rf(self):
        cv_avg = pd.read_csv(f'./results/pca_{0.98}_cv_avg.csv',index_col=0)
        p: Param
        results = []
        for p in self.pg.gen():
            if not self.status.read_key(p,'ok'):
                continue
            counts, test_result = self.status.read_kfold(p)
            print(counts,test_result)
            results.append([p.split_type,
                            f'{p.max_tree_len}_{p.dnn}',
                            test_result['val_acc_epoch'],
                            ])
        cv_dense = pd.DataFrame(results, columns=['data','tree_dnn','dense_test'])
        cv_all = pd.merge(cv_dense,cv_avg,on=['data','tree_dnn'])
        #print(cv_all)
        #print(cv_avg)
        for data, data_res in cv_all.groupby('data'):
            melted = pd.melt(data_res[['tree_dnn','dense_test','svm_test','rf_test']],id_vars='tree_dnn',var_name='classifier',value_name='accuracy')
            melted = melted.sort_values(['tree_dnn'],ascending=True)
            print(melted)
            fig, ax = plt.subplots(figsize=(7,6))
            sns.barplot(x='tree_dnn',y='accuracy',hue='classifier',data=melted,ax=ax)
            ax.set_ylim(0.5,0.9)
            fn = f'barcatplot_all_{data}.png'
            path = os.path.join('./figure',fn)
            print(f'save {path}')
            plt.savefig(path)
            