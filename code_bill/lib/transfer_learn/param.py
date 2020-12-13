from dataclasses import dataclass, asdict
from lib.settings.config import settings
import itertools

__all__ = ['Param', 'ParamGenerator']


@dataclass(eq=True, frozen=True)
class Param():
    exp: int
    classifier: str
    reduction: str
    freeze_type: str
    pretrain_model: str
    split_type: str
    tree: str
    max_tree_len: int
    limit: int
    dnn: str
    auxiliary: bool

    @property
    def experiment_name(self):
        ret = f'exp={self.exp}-cf={self.classifier}-rd={self.reduction}-st={self.split_type}-t={self.tree}-m={self.max_tree_len}-d={self.dnn}-a={self.auxiliary}'
        return ret

class ParamGenerator():
    def __init__(self):
        p = settings.transfer.param
        self.l = []
        pm = Param.__annotations__.keys()
        for k in pm:
            self.l.append(list(p[k]))

    def gen(self):
        status_record = set()
        for i in itertools.product(*self.l):
            dp = asdict(Param(*i))
            
            if dp['tree'] == 'none':

                dp['dnn'] = 'none'
                dp['auxiliary'] = False
                dp['max_tree_len'] = 0

            elif dp['tree'] == 'node2vec' and dp['dnn'] == 'LSTM':
                continue
            
            p = Param(**dp)
            if p in status_record:
                continue
            else:
                status_record.add(p)
            
            yield p
