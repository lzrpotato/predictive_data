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
    tree: bool
    max_tree_len: int

class ParamGenerator():
    def __init__(self):
        p = settings.transfer.param
        self.l = []
        pm = Param.__annotations__.keys()
        for k in pm:
            self.l.append(list(p[k]))

    def gen(self):
        for i in itertools.product(*self.l):
            p = Param(*i)
            if not p.tree:
                p.max_tree_len = 0
            yield p
