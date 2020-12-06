from dataclasses import dataclass
from lib.settings.config import settings
import itertools

__all__ = ['Param', 'ParamGenerator']


@dataclass
class Param():
    exp: int
    layer_num: int
    freeze_type: str
    pretrain_model: str
    split_type: str
    tree: str
    max_tree_len: int
    limit: int
    dnn: str

class ParamGenerator():
    def __init__(self):
        p = settings.transfer.param
        self.l = []
        pm = Param.__annotations__.keys()
        for k in pm:
            self.l.append(list(p[k]))

    def gen(self):
        flag_1 = True
        for i in itertools.product(*self.l):
            p = Param(*i)
            
            if p.tree == 'none':
                if flag_1:
                    p.dnn = 'none'
                    p.max_tree_len = 0
                    flag_1 = False
                else:
                    continue
            elif p.tree == 'node2vec' and p.dnn == 'LSTM':
                continue

            yield p
