from lib.utils import TwitterData, CleanData
import pandas as pd
import gensim.downloader as api
from gensim.models import KeyedVectors
from lib.models.bert import BertMNLIFinetuner
import pytorch_lightning as pl

def main():
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    pl.seed_everything(1234)
    model = BertMNLIFinetuner('bert-base-cased')
    model.setup('fit')
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model)

    result = trainer.test(model)
    print(result)
    
def test():
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

if __name__ == '__main__':
    main()