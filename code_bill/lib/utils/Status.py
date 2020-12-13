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
        pdict = asdict(p)
        pdict.update(result)
        print(pdict)
        self.db.insert(pdict)

    def check_state(self, p: Param):
        existed = True
        pq = Query()
        res = self.db.search(
            (pq.classifier==p.classifier) \
            & (pq.reduction==p.reduction) \
            & (pq.freeze_type == p.freeze_type) \
            & (pq.pretrain_model == p.pretrain_model) \
            & (pq.split_type == p.split_type) \
            & (pq.tree == p.tree) \
            & (pq.max_tree_len == p.max_tree_len) \
            & (pq.limit == p.limit) \
            & (pq.dnn == p.dnn) \
            & (pq.auxiliary == p.auxiliary) \
        )

        if res == []:
            existed = False

        return existed

    def read_best_results(self, p: Param):
        pq = Query()
        res = self.db.search(
            (pq.classifier==p.classifier) \
            & (pq.reduction==p.reduction) \
            & (pq.freeze_type == p.freeze_type) \
            & (pq.pretrain_model == p.pretrain_model) \
            & (pq.split_type == p.split_type) \
            & (pq.tree == p.tree) \
            & (pq.max_tree_len == p.max_tree_len) \
            & (pq.limit == p.limit) \
            & (pq.dnn == p.dnn) \
            & (pq.auxiliary == p.auxiliary) \
        )

        counts = len(res)
        return counts, res[0]['test_acc_epoch']
    
    def read_key(self, p, key):
        pq = Query()
        res = self.db.search(
            (pq.classifier==p.classifier) \
            & (pq.reduction==p.reduction) \
            & (pq.freeze_type == p.freeze_type) \
            & (pq.pretrain_model == p.pretrain_model) \
            & (pq.split_type == p.split_type) \
            & (pq.tree == p.tree) \
            & (pq.max_tree_len == p.max_tree_len) \
            & (pq.limit == p.limit) \
            & (pq.dnn == p.dnn) \
            & (pq.auxiliary == p.auxiliary) \
        )
        counts = len(res)
        if counts == 0:
            return None
        if key in res[0].keys():
            return res[0][key]

    def read_kfold(self, p: Param):
        pq = Query()
        res = self.db.search(
            (pq.classifier==p.classifier) \
            & (pq.reduction==p.reduction) \
            & (pq.freeze_type == p.freeze_type) \
            & (pq.pretrain_model == p.pretrain_model) \
            & (pq.split_type == p.split_type) \
            & (pq.tree == p.tree) \
            & (pq.max_tree_len == p.max_tree_len) \
            & (pq.limit == p.limit) \
            & (pq.dnn == p.dnn) \
            & (pq.auxiliary == p.auxiliary) \
        )

        counts = len(res)
        metrics = ['test_acc_epoch','val_acc_epoch',
                    'precision','recall','fscore','support']
        rets = {}
        if counts == 0:
            return counts, None
        for me in metrics:
            if me not in res[0]:
                continue
            rets[me] = res[0][me]
        
        return counts, rets