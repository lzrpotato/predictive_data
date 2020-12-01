import os
from dataclasses import asdict

import numpy as np
import pytorch_lightning as pl
from lib.models.bert import BertMNLIFinetuner
from lib.settings.config import settings
from lib.transfer_learn.param import Param, ParamGenerator
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CometLogger
from tinydb import Query, TinyDB

__all__ = ['TransferFactory']


class Status():
    def __init__(self):
        self.dbpath = settings.checkpoint + settings.transfer.dbname
        self.db = TinyDB(self.dbpath, sort_keys=True,indent='\t',separators=(',',': '))

    def save_state(self, p, result):
        metrics = ['train_acc_epoch','val_acc_epoch','test_acc_epoch']
        nr = {}
        for metric in metrics:
            nr[metric] = float(result[metric])
        
        nr.update(asdict(p))
        print(nr)
        self.db.insert(nr)

    def check_state(self, p: Param):
        existed = True
        pq = Query()
        res = self.db.search(
            (pq.layer_num==p.layer_num) \
            & (pq.freeze_type == p.freeze_type) \
            & (pq.pretrain_model == p.pretrain_model) \
            & (pq.split_type == p.split_type) \
            & (pq.tree == p.tree) \
            & (pq.max_tree_len == p.max_tree_len)
        )
        
        if res == []:
            existed = False
        
        return existed


class TransferFactory():
    def __init__(self):
        self.pg = ParamGenerator()
        self.status = Status()
        
    def set_config(self, p: Param):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        pl.seed_everything(1234)

        es_cb = EarlyStopping(
            monitor='val_loss_epoch',
            patience=15,
            mode='min',
        )
        
        ckp_cb = ModelCheckpoint(
            monitor='val_loss_epoch',
            dirpath=settings.checkpoint,
            filename='bert-best-model-{epoch:02d}-{train_loss:.2f}',
            save_top_k=1,
            mode='min'
        )
        self.ckp_cb = ckp_cb
        
        comet_logger = CometLogger(
            api_key='RHywDLGIc61n40dBpkSqcmqp7',
            project_name='fakenews',  # Optional
            workspace='lzrpotato',
            experiment_name=f'{p.split_type}-{p.tree}-{p.max_tree_len}-{p.layer_num}',  # Optional
            offline=False,
        )

        lr_monitor = LearningRateMonitor(
            logging_interval='epoch',
        )

        self.trainer = pl.Trainer(gpus=1, 
                            max_epochs=50,
                            progress_bar_refresh_rate=1,
                            flush_logs_every_n_steps=100,
                            callbacks=[es_cb,ckp_cb,lr_monitor],
                            logger=comet_logger
                            )

    def run(self):
        for p in self.pg.gen():
            print(p)
            if self.check_state(p):
                continue
            
            model_type ='DNN'
            if model_type == 'DNN':
                self.set_config(p)
                model = BertMNLIFinetuner(
                    p.pretrain_model,
                    layer_num=p.layer_num,
                    tree=p.tree,
                    max_tree_length=p.max_tree_len,
                    freeze_type=p.freeze_type,
                    split_type=p.split_type,
                )
                model.setup('fit')
                
                self.trainer.fit(model)
                print('load bm from checkpoint: ',self.ckp_cb.best_model_path)
                model.load_from_checkpoint(self.ckp_cb.best_model_path)
                test_result = self.trainer.test(model)
                
                print('rm file', self.ckp_cb.best_model_path)
                os.remove(self.ckp_cb.best_model_path)

                self.save_state(p, test_result[0])
            elif model_type == 'SVM':
                from lib.utils.twitter_data import TwitterData
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.svm import SVC
                twdata = TwitterData(settings.data, p.pretrain_model,tree=p.tree,split_type=p.split_type,max_tree_length=p.max_tree_len)
                twdata.setup()
                train_x = np.array([tree.numpy() for _,_,_,tree,_ in twdata.dataset['train']])
                val_x = np.array([tree.numpy() for _,_,_,tree,_ in twdata.dataset['val']])
                train_y = np.array([label.numpy() for _,_,_,_,label in twdata.dataset['train']])
                val_y = np.array([label.numpy() for _,_,_,_,label in twdata.dataset['val']])
                
                clf = RandomForestClassifier()
                clf.fit(train_x,train_y)
                train_score = clf.score(train_x,train_y)
                val_score = clf.score(val_x,val_y)
                print(f'train_score {train_score}, val_score {val_score}')
            
    def save_state(self, p: Param, result: dict):
        self.status.save_state(p, result)

    def check_state(self, p: Param):
        return self.status.check_state(p)
