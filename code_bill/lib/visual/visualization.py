import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from lib.transfer_learn.param import Param, ParamGenerator
from lib.utils.Status import Status

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
            p.tree
            results.append([p.dnn,p.max_tree_len,p.tree,test_result])

        data = pd.DataFrame(results, columns=['dnn','maxlen','tree','test'])
        #data = data.sort_values(by=['tree','maxlen','dnn'])
        print(data)
        #for tree
        #for dnn, dnn_rst in data.groupby('dnn'):
        #    for 

