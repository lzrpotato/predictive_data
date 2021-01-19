import sys
sys.path.append('.')

import os
from lib.utils.twitter_data import TwitterData
import matplotlib.pyplot as plt
import numpy as np

if not os.path.isdir('./propagation_tree_img/'):
    os.mkdir('./propagation_tree_img/')

def plot_image(ax,tree,lb):
    n = int(np.sqrt(mll))
    img = np.resize(tree,(n,n))
    
    ax[lb//2,lb%2].imshow(img)
    ax[lb//2,lb%2].set_title(f'label {lb}')

def plot_line(ax,tree,lb):
    ax[lb//2,lb%2].plot(range(len(tree)),tree)
    ax[lb//2,lb%2].set_title(f'label {lb}')

for mll in range(100,1001,100):
    print('max tree length ',mll)
    td = TwitterData(tree='tree',max_tree_length=mll,split_type='15_tv')
    td.setup()

    all_class = {}
    fig, ax = plt.subplots(nrows=2,ncols=2)
    for _,_,_,tree,label in td.train_dataloader:
        for i in range(32):
            lb = int(label[i])
            if len(all_class) == 4:
                break
            if lb not in all_class:
                all_class[lb] = True
            else:
                continue
            
            print('label ',lb)
            t = tree[i].cpu().numpy()
            plot_line(ax,t,lb)

    plt.tight_layout()
    plt.savefig(f'./propagation_tree_img/pt_{mll}.png')
    plt.close()