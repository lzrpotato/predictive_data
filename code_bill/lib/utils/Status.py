from tinydb import TinyDB, Query
from lib.settings.config import settings
from dataclasses import asdict
from lib.transfer_learn.param import Param

class Status():
    def __init__(self,exp=None):
        if exp is None:
            exp = int(settings.transfer.param.exp[0])
        self.dbpath = settings.checkpoint + f'exp={exp}_'+ settings.transfer.dbname
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
