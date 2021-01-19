import os
import sys
sys.path.append('.')
from pprint import pprint
from lib.settings.config import settings
import pickle
from tqdm import tqdm
from anytree import Node, RenderTree, PreOrderIter, LevelOrderIter
import ast
from lib.utils.twitter_data import MyNode
import numpy as np



class Statistics():
    def __init__(self):
        self.root = settings.data

    def Show_Statistics(self):
        self.load_data()

    def load_data(self):
        tw = ['twitter15','twitter16']
        for t in tw:
            label_p = os.path.join(self.root,t,'label.txt')
            source_p = os.path.join(self.root,t,'source_tweets.txt')
            tree_p = os.path.join(self.root,t,'tree')
            labels, class_count = self._read_label(label_p)
            
            tree_map = self._read_tree(t, tree_p)
            summary_tree = self.summary_tree(tree_map)
            print(t)
            pprint(class_count)
            print(summary_tree)
            
    def _read_label(self, path):
        pairs = {}
        class_count = {}
        with open(path, mode='r') as f:
            for line in f:
                label, id = line.split(':')
                if label not in class_count:
                    class_count[label] = 1
                else:
                    class_count[label] += 1
                if id not in pairs.keys():
                    pairs[int(id)] = label
                else:
                    print('error')
        
        return pairs, class_count

    def _read_text(self, path):
        pairs = {}
        with open(path, mode='r') as f:
            for line in f:
                id, text = line.split('\t')
                if id not in pairs.keys():

                    pairs[int(id)] = text
                else:
                    print('error')
        return pairs

    def summary_tree(self, tree_map):
        all_max_time = []
        all_post_num = []
        for id, tree in tree_map.items():
            max_time = -10000
            post_num = 0
            node: Node
            for node in LevelOrderIter(tree):
                post_num += 1
                
                if max_time < node.name.t:
                    max_time = node.name.t
            
            all_max_time.append(max_time)
            all_post_num.append(post_num)

        avg_time = np.mean(all_max_time)/60
        avg_n_posts = np.mean(all_post_num)
        max_n_posts = np.max(all_post_num)
        min_n_posts = np.min(all_post_num)
        return {'avg_time': avg_time, 'avg_post':avg_n_posts,
                'max_posts': max_n_posts, 'min_posts': min_n_posts
                }

    def _read_tree(self, t, path):
        pickle_fn = f"tree_maps_{t}.p"
        if os.path.isfile(os.path.join(settings.checkpoint,pickle_fn)):
            tree_map = pickle.load(open(os.path.join(settings.checkpoint,pickle_fn), "rb" ))
            print(f'load {pickle_fn}')
            return tree_map
        
        tree_map = {}
        for fn in tqdm(os.listdir(path)):
            index = fn.split('.')[0]
            tree_map[int(index)] = self._build_tree(os.path.join(path,fn))
        
        pickle.dump(tree_map, open(os.path.join(settings.checkpoint,pickle_fn), "wb"))
        print(f'saved {pickle_fn}')
        return tree_map

    def _build_tree(self, fn):
        root = None
        nodemap = {}
        with open(fn, mode='r') as f:
            for line in f:
                splited = line.split('->')
                p = ast.literal_eval(splited[0])
                c = ast.literal_eval(splited[1])
                np = Node(MyNode(*p))
                
                if root is None and np.name.id == 0:
                    root = Node(MyNode(*c))
                    nodemap[root.name.id] = root
                    continue
                    
                if np.name.id not in nodemap:
                    nodemap[np.name.id] = np
                myp = nodemap[np.name.id]

                nc = Node(MyNode(*c), parent=myp)
                nodemap[nc.name.id] = nc
        
        return root

if __name__ == '__main__':
    ss = Statistics()
    ss.Show_Statistics()