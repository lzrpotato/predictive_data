import os
import uuid

import lxml.etree as et

from .log import setup_custom_logger

logger = setup_custom_logger('ResourceManager', __file__)


class ResourceManager():
    def __init__(self, path: str, fPath: str, name: str, resType: str):
        self._path = path
        self._name = name
        self._fPath = fPath
        self._xmltree = None
        self._resm_fn = self._name+'.rsm.xml'
        self._loadConfig()
        self._temp_registered = {}

    def _loadConfig(self):
        if not os.path.isdir(self._path):
            os.mkdir(self._path)

        if not os.path.isfile(self._path+self._resm_fn):
            root = et.Element('ResouceManager')
            root.set('namespace', self._name)
            root.set('fpath', self._fPath)
            self._xmltree = et.ElementTree(root)
            self._saveConfig()
        else:
            parser = et.XMLParser(remove_blank_text=True)
            self._xmltree = et.parse(self._path+self._resm_fn, parser)

    def _saveConfig(self):
        if self._xmltree != None:
            self._xmltree.write(self._path+self._resm_fn, pretty_print=True)
        else:
            logger.warn(f'[_saveConfig] try to save uninitalized config!')

    def _getElement(self, search_dict):
        root = self._xmltree.getroot()
        elem = None
        for e in root:
            not_match = False
            size_e = len(e.attrib)
            size_sd = 0

            for k, v in search_dict.items():
                if k[0] != '#':
                    size_sd += 1
                    if k in e.attrib and e.attrib[k] == str(v):
                        continue
                    else:
                        not_match = True
                        break

            if not_match == False and size_e == size_sd:
                #logger.debug(f' {e} {not_match} e {size_e} sd {size_sd}')
                elem = e
                return elem

        return None

    def _createElement(self, search_dict, ID):
        newelem = et.Element(self._name)
        keys = list(search_dict)
        keys.sort()
        for k in keys:
            if k[0] != '#':
                newelem.set(k, str(search_dict[k]))
        #newelem.text = ID
        e_id = et.SubElement(newelem, 'uuid')
        e_id.text = ID
        for k in keys:
            if k[0] == '#':
                e_k = et.SubElement(newelem, k[1:])
                e_k.text = str(search_dict[k])

        return newelem

    def search(self, search_dict: dict, keys=None) -> str:
        e = self._getElement(search_dict)
        if e == None:
            return ''

        if keys == None:
            return e.find('uuid').text
        else:
            result = ''
            if isinstance(keys, str):
                if e.find(keys[1:]) == None:
                    result = ''
                else:
                    result = e.find(keys[1:]).text
                return result

            results = []
            for k in keys:
                if k[0] == '#':
                    if e.find(k[1:]) == None:
                        results.append('')
                    else:
                        results.append(e.find(k[1:]).text)
            return tuple(results)

    def register(self, search_dict: dict):
        temp_id = ''
        if '#uuid' in search_dict.keys():
            temp_id = search_dict['#uuid']

        e = self._getElement(search_dict)
        if e == None:
            if temp_id == '':
                temp_id = str(uuid.uuid4())
            # register to temp list, and mark it as appending later
            self._temp_registered[temp_id] = [search_dict, 'a']
        else:
            temp_id = e.find('uuid').text
            # find old entry, mark it as replace later
            self._temp_registered[temp_id] = [search_dict, 'r']

        return temp_id

    def commit(self, ID: str):

        if ID in self._temp_registered.keys():
            sd, mode = self._temp_registered[ID]

            if mode == 'a':
                root = self._xmltree.getroot()
                newelem = self._createElement(sd, ID)
                root.append(newelem)
            elif mode == 'r':
                e = self._getElement(sd)
                e.find('uuid').text = ID

                for k in sd.keys():
                    if k[0] == '#':
                        if e.find(k[1:]) != None:
                            e.find(k[1:]).text = str(sd[k])
                        else:
                            sub_e = et.SubElement(e, k[1:])
                            sub_e.text = str(sd[k])

            self._saveConfig()
            self._temp_registered.pop(ID)
        else:
            logger.warn(f'[commit] {ID} not registered')
            logger.warn(f'[commit] temp {self._temp_registered.keys()}')

    def res_gen(self, keys):
        for model in self._xmltree.getroot():
            res = [model.attrib]
            for entry in model:
                if entry.tag in keys:
                    res.append(entry.text)
            yield res


def test():
    rm = ResourceManager('./checkpoint/', './BestModel/', 'BestModel',
                         'Model')

    # id = rm.register({'name':'vgg16','train':'all','freeze':'n','#acc':0.99})
    # id = rm.register({'name':'vgg16','train':'all','freeze':'n','#trace':'yes'})
    # print(id)
    # rm.commit(id)
    search_dict = {
        'name': 'vgg16',
        'nofze': 'True' if True else 'False',
        'epoch': str(1),
        'trainall': 'True' if False else 'False',
        '#model': 'True'
    }
    #myuuid = rm.register(search_dict)
    # rm.commit(myuuid)
    # logger.debug('hi')
    search_dict = {
        'name': 'vgg16',
        'nofze': 'True' if True else 'False',
        'epoch': str(10),
        'trainall': 'True' if False else 'False',
        '#trace': 'True'
    }
    # myuuid = rm.search(search_dict,'#trace')
    # if myuuid == '':
    #myuuid = rm.register(search_dict)
    # rm.commit(myuuid)
    print(rm.search(search_dict))
    print(rm.search(search_dict, ['#model']))


def test1():
    rm = ResourceManager('./checkpoint/', './BestModel/', 'BestModel',
                         'Model')
    search_dict = {'name': 'vgg16', 'nofze': 'True',
                   'epoch': '100', 'trainall': 'False', 'binary': 'True'}
    rs = rm.search(search_dict, '#model')
    print(rs)
    if rs == 'True':
        print(rs)


def test_gen():
    import pandas as pd
    rm = ResourceManager('./checkpoint/', './BestModel/', 'BestModel',
                         'Model')
    l = []
    for mydic, myuid in rm.res_gen(['uuid']):
        fn = './results/' + 'la_' + myuid + '.csv'
        la = pd.read_csv(fn, index_col=0)
        l.append([mydic['name'], mydic['binary'], *
                  list(la.iloc[-8, [1, 3]]), la.shape[0]-7])

    summary = pd.DataFrame(
        l, columns=['model', 'binary', 'train_acc', 'test_acc', 'epoch'])
    import numpy as np
    print(np.corrcoef(summary['train_acc'], summary['epoch']))
    print(summary)


if __name__ == '__main__':
    test_gen()
