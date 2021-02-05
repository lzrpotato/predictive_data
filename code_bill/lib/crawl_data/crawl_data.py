from twitter_bot import Twitterbot, get_credentials
from concurrent.futures import ThreadPoolExecutor
import time
import tqdm
import os
import ast
from anytree import Node, RenderTree, PreOrderIter, LevelOrderIter
from dataclasses import dataclass

@dataclass
class MyNode():
    id: int
    sid: int
    t: float
    def __init__(self, id, sid, t):
        if id == 'ROOT': 
            self.id = 0
        else:
            self.id = int(id)
        if sid == 'ROOT':
            self.sid = 0
        else:
            self.sid = int(sid)
        self.t = float(t)

    def __repr__(self):
        return str(self.sid) + '_' + str(self.t)

class CrawlData():
    def __init__(self):
        #get_credentials('./credentials.txt')
        pass

    def task_get_content_profile(self, id):
        bot = Twitterbot()
        content, username, tweet_time = bot.find_tweet_by_id(id)
        profile = bot.find_profile_by_name(username)
        #print(content,username,tweet_time,profile)
        bot.close()
        return content,username,tweet_time,profile

    def read_tree(self, path):
        tree_map = {}
        for fn in tqdm(os.listdir(path)):
            index = fn.split('.')[0]
            tree_map[int(index)] = self._build_tree(os.path.join(path,fn))

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

    def crawl_retweet_threading(self):
        ids = ['1339363006854037504','624298742162845696']
        start_time = time.time()
        with ThreadPoolExecutor(max_workers = 2) as executor:
            results = executor.map(self.task_get_content_profile, ids)
        
        for result in results:
            print(result)
        end_time = time.time()
        print('crawl time ', end_time-start_time)

    def crawl_retweet(self):
        ids = ['1339363006854037504','624298742162845696']
        start_time = time.time()
        results = []
        for id in ids:
            results.append(self.task_get_content_profile(id))
        
        for result in results:
            print(result)
        end_time = time.time()
        print('crawl time ', end_time-start_time)

    def crawl_user_profile(self):
        pass


if __name__ == '__main__':
    crawl_data = CrawlData()
    crawl_data.crawl_retweet_threading()
    #crawl_data.crawl_retweet()