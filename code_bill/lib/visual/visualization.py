import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from lib.transfer_learn.param import Param, ParamGenerator
from lib.utils.Status import Status
import os
import numpy as np

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
        cv_avg.columns = ['data','tree_dnn','SVM','RF']
        p: Param
        results = []
        for p in self.pg.gen():
            if not self.status.read_key(p,'ok'):
                continue
            counts, test_result = self.status.read_kfold(p)
            results.append([p.split_type,
                            f'{p.max_tree_len}_{p.dnn}',
                            test_result['val_acc_epoch'],
                            ])
        cv_dense = pd.DataFrame(results, columns=['data','tree_dnn','Dense'])
        cv_all = pd.merge(cv_dense,cv_avg,on=['data','tree_dnn'])
        #print(cv_all)
        #print(cv_avg)
        for data, data_res in cv_all.groupby('data'):
            melted = pd.melt(data_res[['tree_dnn','Dense','SVM','RF']],id_vars='tree_dnn',var_name='Classifier',value_name='Accuracy')
            
            melted = melted.sort_values(['tree_dnn'],ascending=True)
            #print(melted['tree_dnn'].str.split('_')[1] == 'LSTM')
            #print(melted[melted['tree_dnn'].str.contains('none|CNN')])
            #melted = melted[melted['tree_dnn'].str.contains('none|CNN')]
            fig, ax = plt.subplots(figsize=(12,6))
            g = sns.barplot(x='tree_dnn',y='Accuracy',hue='Classifier',data=melted,ax=ax)
            ax.set_ylim(0.5,0.9)
            ax.set_xlabel('max length of tree & add-on model')
            ax.set_title('Accuracy against max length of tree and add-on model')
            def show_values_on_bars(axs):
                def _show_on_single_plot(ax):        
                    for p in ax.patches:
                        _x = p.get_x() + p.get_width() / 2
                        _y = p.get_y() + p.get_height()
                        value = '{:.2f}'.format(p.get_height())
                        ax.text(_x, _y, value, ha="center") 

                if isinstance(axs, np.ndarray):
                    for idx, ax in np.ndenumerate(axs):
                        _show_on_single_plot(ax)
                else:
                    _show_on_single_plot(axs)
            show_values_on_bars(ax)
            fn = f'barcatplot_all_{data}.png'
            path = os.path.join('./figure',fn)
            print(f'save {path}')
            plt.savefig(path)
            