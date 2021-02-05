from dataclasses import dataclass, asdict
from logging import makeLogRecord
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

    def __init__(
        self,
        exp,
        classifier,
        reduction,
        freeze_type,
        pretrain_model,
        split_type,
        tree,
        max_tree_len,
        limit,
        dnn,
        auxiliary):

        if not isinstance(exp,int):
            raise ValueError(f'exp shoule be int, but "{exp}"')
        object.__setattr__(self, 'exp', exp)

        if classifier not in ['dense_1']:
            raise ValueError(f'classifier shoule be dense_1, but "{classifier}"')
        object.__setattr__(self, 'classifier', classifier)

        if reduction not in ['pca']:
            raise ValueError(f'reduction shoule be pca, but "{reduction}"')
        object.__setattr__(self, 'reduction', reduction)

        if freeze_type not in ['no', 'yes']:
            raise ValueError(f'freeze_type shoule be no or yes, but "{freeze_type}"')
        object.__setattr__(self, 'freeze_type', freeze_type)

        if pretrain_model not in ['bert-base-cased','roberta-base','bert-base-uncased']:
            raise ValueError(f'pretrain_model shoule be bert-base-cased, but "{pretrain_model}"')
        object.__setattr__(self, 'pretrain_model', pretrain_model)

        if split_type not in ['15_tv','16_tv']:
            raise ValueError(f'split_type shoule be 15_tv, but "{split_type}"')
        object.__setattr__(self, 'split_type', split_type)

        if tree not in ['tree','none']:
            raise ValueError(f'freeze_type shoule be tree, but "{tree}"')
        object.__setattr__(self, 'tree', tree)

        if not isinstance(max_tree_len,int):
            raise ValueError(f'max_tree_len shoule be no or yes, but "{max_tree_len}"')
        object.__setattr__(self, 'max_tree_len', max_tree_len)

        if not isinstance(limit,int):
            raise ValueError(f'limit shoule be int, but "{limit}"')
        object.__setattr__(self, 'limit', limit)

        d = dnn.split('_')[0]
        if d not in ['CNN','CNNOri','PTCNN','FCN','CNNAVG','CNNFIX','CNNRes','CNNDEP','CNNTK','CNNOK','CNNTKF']:
            raise ValueError(f'dnn shoule be CNN, but "{dnn}"')
        object.__setattr__(self, 'dnn', dnn)

        if not isinstance(auxiliary,bool):
            raise ValueError(f'freeze_type shoule be no or yes, but "{auxiliary}"')
        object.__setattr__(self, 'auxiliary', auxiliary)

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
