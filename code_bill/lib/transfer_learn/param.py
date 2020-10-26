from dataclasses import dataclass
from tinydb import TinyDB, Query
from lib.settings.config import settings
import itertools

__all__ = ['Param', 'ParamGenerator']


@dataclass
class Param():
    layer_num: int
    freeze_type: str
    pretrain_model: str
    split_type: str

class ParamGenerator():
    def __init__(self):
        p = settings.transfer.param
        self.l = [p.layer_num,p.freeze_type,p.pretrain_model,p.split_type]

    def gen(self):
        for i in itertools.product(*self.l):
            p = Param(i[0],i[1],i[2],i[3])
            yield p
