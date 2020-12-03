import operator 
from tqdm import tqdm
import re

__all__ = ['CleanData']


class CleanData():
    def __init__(self):
        pass
    
    def build_vocab(self, sentences, verbose=True):
        vocab = {}
        for sentence in tqdm(sentences, disable = (not verbose)):
            for word in sentence:
                try:
                    vocab[word] += 1
                except KeyError:
                    vocab[word] = 1
        return vocab
    
    def check_coverage(self, vocab, embeddings_index):
        a = {}
        oov = {}
        k = 0
        i = 0
        for word in tqdm(vocab):
            try:
                a[word] = embeddings_index[word]
                k += vocab[word]
            except:

                oov[word] = vocab[word]
                i += vocab[word]
                pass

        print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
        print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
        sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

        return sorted_x
        
    def clean_text(self, x):

        x = str(x)
        for punct in ["'s","'m", "n't"]:
            x = x.replace(punct, f' {punct} ')
        for punct in "/-'":
            x = x.replace(punct, ' ')
        for punct in '&?!':
            x = x.replace(punct, f' {punct} ')
        for punct in '.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
            x = x.replace(punct, '')
        return x


    def clean_numbers(self, x):

        x = re.sub('[0-9]{5,}', ' ##### ', x)
        x = re.sub('[0-9]{4}', ' #### ', x)
        x = re.sub('[0-9]{3}', ' ### ', x)
        x = re.sub('[0-9]{2}', ' ## ', x)
        x = re.sub('[0-9]{1}', ' # ', x)
        return x
        
    
    def replace_typical_misspell(self, text):
        mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
        mispell_dict = { 'URL':'urls',
                'birdeater':'bird easter',
                'charliehebdo':'charlie hebdo',
                'berniesanders': 'bernie sanders',
                '—potus': 'potus',
                'realdonaldtrump':'donald trump',
                'sydneysiege': 'sydney siege',
                'hillaryclinton': 'hillary clinton',
                'kobane': 'city of syria',
                'mikebrown': 'mike brown',
                #'mh': 'malaysia airlines flight',
                'amymek': 'amy mek',
                'ottawashooting': 'ottawa shooting',
                'michaelbrown': 'michael brown',
                
                }
        
        def _get_mispell(mispell_dict):
            mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
            return mispell_dict, mispell_re
        
        mispellings, mispellings_re = _get_mispell(mispell_dict)
        
        def replace(match):
            return mispellings[match.group(0)]

        return mispellings_re.sub(replace, text)