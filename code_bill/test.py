import pandas as pd
import os

def test_tf():
    from lib.transfer_learn.transfer_factory import TransferFactory
    tf = TransferFactory()
    tf.run()

def test():
    from gensim.models import KeyedVectors

    dataset = TwitterData('../../rumor_detection_acl2017')
    cdf = CleanData()
    build_vab = cdf.build_vocab
    tw15_X, tw15_y, tw16_X, tw16_y = dataset.load_data()
    tw15_X = pd.Series(tw15_X) 
    tw16_X = pd.Series(tw15_X)

    #print({k: vocab[k] for k in list(vocab)[:5]})
    
    #model = api.load('glove-twitter-25')
    #model.wv.save('glove-twitter-25-wv')
    
    wv_form_text = KeyedVectors.load('glove-twitter-25-wv')
    
    tw15_X = tw15_X.apply(lambda x: cdf.clean_text(x))
    tw15_X = tw15_X.apply(lambda x: cdf.clean_numbers(x))
    tw15_X = tw15_X.apply(lambda x: cdf.replace_typical_misspell(x))
    sentences = tw15_X.apply(lambda x: x.split())
    vocab = build_vab(sentences)
    oov = cdf.check_coverage(vocab,wv_form_text)
    print(oov[:20])

def tokenizer_test():
    from transformers import AutoTokenizer
    import numpy as np
    pretrain_tokenizer_model = 'bert-base-cased'

    dataset = TwitterData('../../rumor_detection_acl2017',pretrain_tokenizer_model)
    dataset.prepare_data()
    dataset.setup()
    tokenizer = AutoTokenizer.from_pretrained(pretrain_tokenizer_model, use_fast=True)
    
    batch = tokenizer(list(dataset.tw15_X))
    count = 0
    for i, ids in enumerate(batch['input_ids']):
        unc =  np.count_nonzero(np.array(ids) == 100)
        if unc != 0:
            print(ids)
            print(dataset.tw15_X[i])
            print(tokenizer.decode(ids))
            print('****')
        count += unc
    print(count)

def test_twdata():
    from lib.utils.twitter_data import TwitterData
    pretrain_tokenizer_model = 'bert-base-cased'
    pretrain_model = 'bert-base-cased'
    split_type = '15_tv'
    tree = 'node2vec'
    max_tree_len = [100,500]
    limit = [100]

    for l in limit:
        for mtl in max_tree_len:
            td = TwitterData(tree='tree',max_tree_length=100,datatype='all',split_type=split_type, cv=True)
            td.setup()
            next(td.kfold_gen())
            print(td.train_data[1].shape)

def test_kfold_deterministic():
    from lib.utils.twitter_data import TwitterData
    td = TwitterData(tree='tree',max_tree_length=100,datatype='all',split_type='15_tv', cv=True, kfold_deterministic=True)
    td.setup()
    td.kfold_get_by_fold(1)
    print(td.train_data[1].shape)

def test_summary():
    from lib.utils.data_summary import Statistics
    ss = Statistics()
    ss.Show_Statistics()

def test_slurm_var():
    print('SLURM_JOB_ID ',os.environ.get('SLURM_JOB_ID'))
    print('SLURM_JOB_NAME ',os.environ.get('SLURM_JOB_NAME'))
    print('SLURM_SUBMIT_DIR ',os.environ.get('SLURM_SUBMIT_DIR'))
    print('SLURM_JOB_NODELIST ',os.environ.get('SLURM_JOB_NODELIST'))

if __name__ == '__main__':
    #test_tf()
    test_slurm_var()
    #tokenizer_test()
