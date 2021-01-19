import os

import warnings
from dataclasses import asdict
import numpy as np
import pytorch_lightning as pl
from lib.models.bert import BertMNLIFinetuner
from lib.models.roberta import RoBERTaFinetuner
from lib.settings.config import settings
from lib.transfer_learn.param import Param, ParamGenerator
from lib.utils.twitter_data import TwitterData
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CometLogger
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lib.utils.status_sqlite_bert import Status
__all__ = ['TransferFactory']


class TransferFactory():
    def __init__(self):
        self.pg = ParamGenerator()
        exp = int(list(settings.transfer.param.exp)[0])
        self.status = Status()
        
    def set_config(self, p: Param, fold):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        pl.seed_everything(1234)
        monitor_key = 'acc'
        if monitor_key == 'loss':
            monitor_mode = 'min'
        elif monitor_key == 'acc':
            monitor_mode = 'max'
        else:
            raise ValueError(f'monitor_key "{monitor_key}" is incorrect')
        
        es_cb = EarlyStopping(
            monitor=f'val_{monitor_key}_epoch',
            patience=20,
            mode=monitor_mode,
        )
        model_name = p.pretrain_model.split('-')[0]
        ckp_cb = ModelCheckpoint(
            monitor=f'val_{monitor_key}_epoch',
            dirpath=settings.checkpoint,
            filename= model_name + '-sp=' + p.split_type + '-maxl=' + str(p.max_tree_len) + '-fold='+ str(fold) + '-{epoch:02d}-{val_acc_epoch:.3f}',
            save_top_k=1,
            mode=monitor_mode
        )
        self.ckp_cb = ckp_cb
        
        """
        comet_logger = CometLogger(
            api_key='RHywDLGIc61n40dBpkSqcmqp7',
            project_name='fakenews',  # Optional
            workspace='lzrpotato',
            experiment_name=p.experiment_name,
            offline=False,
        )
        """
        lr_monitor = LearningRateMonitor(
            logging_interval='epoch',
        )

        self.trainer = pl.Trainer(gpus=1,
                            max_epochs=50,
                            progress_bar_refresh_rate=0.1,
                            flush_logs_every_n_steps=100,
                            callbacks=[es_cb,ckp_cb,lr_monitor],
                            #logger=comet_logger
                            )

    def run(self, p: Param):
        twdata = TwitterData(settings.data, p.pretrain_model,tree=p.tree,split_type=p.split_type,
            max_tree_length=p.max_tree_len,limit=p.limit,cv=True,n_splits=5)
        twdata.setup_kfold()
        results_kfold = []
        for fold in twdata.kfold_gen():
            self.set_config(p,fold)
            model = None

            if p.pretrain_model == 'roberta-base':
                model = RoBERTaFinetuner(ep=p,fold=fold)
            elif p.pretrain_model == 'bert-base-cased':
                model = BertMNLIFinetuner(ep=p,fold=fold)
            else:
                raise ValueError('Incorrect pretrain_model')

            test_result = None
            if p.classifier.split('_')[0] == 'dense':
                self.trainer.fit(model,train_dataloader=twdata.train_dataloader,val_dataloaders=twdata.val_dataloader)
                #model.load_from_checkpoint(self.ckp_cb.best_model_path)
                #model = RoBERTaFinetuner.load_from_checkpoint(self.ckp_cb.best_model_path)
                test_result = self.trainer.test(test_dataloaders=twdata.test_dataloader)
            results_kfold.append(test_result[0])
            print('[test_result]',test_result[0])
            self.save_state(p, test_result[0], fold)
            if os.path.isfile(self.ckp_cb.best_model_path):
                print('rm file', self.ckp_cb.best_model_path)
                os.remove(self.ckp_cb.best_model_path)
            else:
                warnings.warn(f'[Warning] best model path {self.ckp_cb.best_model_path} is incorrect')

        #self.save_state_kfold(p, results_kfold)
        #self.save_state(p, results_kfold[0])

    def save_state(self, p: Param, result: dict, fold):
        res = {'acc':result['test_acc_epoch'],'c1':result['fscore'][1],'c2':result['fscore'][0],'c3':result['fscore'][2],'c4':result['fscore'][3]}
        par = {'dnn':p.dnn,'fold':fold,'SplitType':p.split_type,'MaxLen':p.max_tree_len,'CurEpoch':-1,'ok':True}
        par.update(res)
        self.status.save_status(par)

    def check_state(self, p: Param):
        return self.status.read_status(p)
