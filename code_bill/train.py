import os
os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'
from lib.transfer_learn.param import Param
import argparse
import copy


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected. but {v}')

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default='setting',help='"setting" or "argument"')
    parser.add_argument('--exp',type=int, default=9)
    parser.add_argument('--classifier',type=str, default='dense_1')
    parser.add_argument('--reduction',type=str, default='pca')
    parser.add_argument('--freeze_type',type=str, default='no')
    parser.add_argument('--pretrain_model',type=str, default='bert-base-cased')
    parser.add_argument('--split_type',type=str, default='16_tv')
    parser.add_argument('--tree',type=str, default='tree')
    parser.add_argument('--max_tree_len',type=int, default=300)
    parser.add_argument('--limit',type=int, default=100)
    parser.add_argument('--dnn',type=str, default='CNN')
    parser.add_argument("--auxiliary", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="")
    args = parser.parse_args()
    arg_dict = copy.deepcopy(vars(args))
    arg_dict.pop('mode',None)
    p = Param(**arg_dict)
    
    return args, p

def train():
    print('COMET_MODE=',os.environ['COMET_MODE'])
    from lib.transfer_learn.transfer_factory import TransferFactory
    tf = TransferFactory()
    tf.run()

def train_p(p):
    print('COMET_MODE=',os.environ['COMET_MODE'])
    from lib.transfer_learn.paralle_transfer_factory import TransferFactory
    tf = TransferFactory()
    tf.run(p)

if __name__ == '__main__':
    #train()
    args, p = get_arg()
    if args.mode == 'setting':
        train()
    else:
        train_p(p)