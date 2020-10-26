import seaborn as sns
import matplotlib.pyplot as plt
from tinydb import TinyDB
import pandas as pd
from lib.settings.config import settings

class Visual():
    def __init__(self):
        pass

    def _draw_basic(self, df, key, filename):
        sns.set_style('whitegrid')
                
        f, axes = plt.subplots(1,1,figsize=(5,5))
        ax1 = sns.lineplot(data=df[key],markers=True,linewidth=1.5,ax=axes,legend='brief')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(filename)

    def draw_transfer(self):
        db = TinyDB(settings.checkpoint+settings.transfer.dbname)
        keys = [
            "freeze_type",
			"layer_num",
			"pretrain_model",
			"split_type",
			"train_acc_mean",
            "val_acc_mean",
            "test_acc_mean"
        ]
        data = [[i[k] for k in keys] for i in db.all()]
        data = pd.DataFrame(data, columns=keys)

        for pm, pm_r in data.groupby('pretrain_model'):
            for st, st_r in pm_r.groupby('split_type'):
                newdf = st_r.pivot(index='layer_num',columns='freeze_type',values=['train_acc_mean','val_acc_mean','test_acc_mean'])
                key = 'test_acc_mean'
                filename = settings.fig+f'{pm}_{st}_lf_{key}.png'
                self._draw_basic(newdf, key, filename)
                
                newdf = st_r.pivot(index='freeze_type',columns='layer_num',values=['train_acc_mean','val_acc_mean','test_acc_mean'])
                key = 'test_acc_mean'
                filename = settings.fig+f'{pm}_{st}_fl_{key}.png'
                self._draw_basic(newdf, key, filename)
                