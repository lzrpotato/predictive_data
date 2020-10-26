from lib.models.bert import BertMNLIFinetuner
from tinydb import TinyDB, Query
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from dataclasses import asdict

from lib.settings.config import settings
from lib.transfer_learn.param import Param, ParamGenerator

__all__ = ['TransferFactory']


class Status():
    def __init__(self):
        self.dbpath = settings.checkpoint + settings.transfer.dbname
        self.db = TinyDB(self.dbpath)

    def save_state(self, p, result):
        metrics = ['train_acc_mean','val_acc_mean','test_acc_mean']
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
            & (pq.split_type == p.split_type)
        )
        
        if res == []:
            existed = False
        
        return existed



class TransferFactory():
    def __init__(self):
        #self.set_config()
        self.pg = ParamGenerator()
        self.status = Status()

    def set_config(self):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        pl.seed_everything(1234)

        es_cb = EarlyStopping(
            monitor='train_loss',
            patience=12,
            mode='min',
        )
        ckp_cb = ModelCheckpoint(
            monitor='train_loss',
            filepath= settings.checkpoint + 'bert-best-model-{epoch:02d}-{train_loss:.2f}',
            save_top_k=1,
            mode='min'
        )
        self.ckp_cb = ckp_cb
        self.trainer = pl.Trainer(gpus=1, 
                            max_epochs=50,
                            progress_bar_refresh_rate=20,
                            flush_logs_every_n_steps=100,
                            callbacks=[es_cb],
                            checkpoint_callback=ckp_cb,
                            default_root_dir=settings.checkpoint)


    def run(self):
        for p in self.pg.gen():
            
            if self.check_state(p):
                continue
            
            model = BertMNLIFinetuner(
                    p.pretrain_model,
                    layer_num=p.layer_num,
                    freeze_type=p.freeze_type,
                    split_type=p.split_type,
                )
            
            model.setup('fit')
            self.trainer.fit(model)
            print('load bm from checkpoint: ',self.ckp_cb.best_model_path)
            model.load_from_checkpoint(self.ckp_cb.best_model_path)
            test_result = self.trainer.test(model)
            
            self.save_state(p, test_result[0])
            
    def save_state(self, p: Param, result: dict):
        self.status.save_state(p, result)

    def check_state(self, p: Param):
        self.status.check_state(p)