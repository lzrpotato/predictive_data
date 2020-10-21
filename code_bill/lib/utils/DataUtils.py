import os
from typing import Tuple, Union

import lxml.etree as et
import numpy as np
import pandas as pd
import torch
from pandas.core.arrays.categorical import Ordered
from torch.utils.data.dataset import Subset
from torchvision import datasets
from torchvision.datasets.folder import ImageFolder

from .ConfigManager import *
from .log import setup_custom_logger

logger = setup_custom_logger('ResourceManager', __file__)


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).
    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        return self.map(self.dataset[index][0]), self.dataset[index][1], index

    def __len__(self):
        return len(self.dataset)


class Subclass(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, subclasses, binary):
        self.conf_fn = 'datamap.conf'
        self.dir_checkpoint = './checkpoint/'
        self.conf_path = self.dir_checkpoint + self.conf_fn
        self.dataset = dataset
        if not isinstance(subclasses, list):
            raise TypeError
        self.subclasses = subclasses
        self.binary = binary
        self.datamap = self._loadDataMap()
        self._maplabel()
        self._make_indices()

    def _maplabel(self):
        self.labelmap = {}
        for i, l in enumerate(self.subclasses):
            if self.binary == True:
                self.labelmap[l] = self.datamap.loc[i, 'binary_i']
            else:
                self.labelmap[l] = i
        #logger.debug(f'[labelmap] {self.labelmap}')
        #self.classes = list(range(len(self.labelmap)))
        if self.binary == True:
            self.classes = np.array(['healthy', 'unhealthy'])
        else:
            self.classes = np.array(self.dataset.classes)[self.subclasses]
        #logger.debug(f'[_maplabel] {self.class_names}')

    def _make_indices(self):
        indices = torch.arange(0, len(self.dataset))
        newindices = None
        for _, rows in self.datamap.iterrows():
            start, end = rows['start'], rows['end']
            if newindices == None:
                newindices = indices[start:end+1]
            else:
                newindices = torch.cat((newindices, indices[start:end+1]), 0)

        self.dataset = Subset(self.dataset, newindices)

    def _loadDataMap(self):
        datamap = DataMapBuilder().readDatamap()
        datamap = datamap[['label', 'binary_i', 'count', 'start', 'end']]
        return datamap.loc[self.subclasses]

    def _convert_label(self, label):
        # if label not in self.labelmap.keys():
        #raise StopIteration
        return self.labelmap[label]

    def __getitem__(self, index):
        #index = self.indices[index]
        return self.dataset[index][0], self._convert_label(self.dataset[index][1])

    def __len__(self):
        return self.datamap['count'].sum()


class DatasetPrepare():
    def __init__(self, dir_data, tf, ratio=0.8, shuffle=True, subclass=[0, 1, 2, 3], binary=False, save_mode=True, ordered=False):
        self.dir_data = dir_data
        self.tf = tf
        self.ratio = ratio
        self.shuffle = shuffle
        self.subclass = subclass
        self.binary = binary
        self.save_mode = save_mode
        self.ordered = ordered
        self.dir_checkpoint = './checkpoint/'
        self._getTrainTestDataset()

    def _getTrainTestDataset(self):
        image_datasets = datasets.ImageFolder(self.dir_data)
        if self.ordered == True:
            def sort_key(k): return int(k[0].split('_')[-1][0:-4])
            image_datasets.samples = sorted(
                image_datasets.samples, key=sort_key)
            image_datasets.imgs = image_datasets.samples
            self.image_datasets = image_datasets

        if self.subclass != None:
            logger.debug(f'[getDataloader] subclasses {self.subclass}')
            image_datasets = Subclass(
                image_datasets, self.subclass, self.binary)

        # set class names
        self.class_names = image_datasets.classes

        # set number of classes
        self.n_class = len(self.class_names)

        # set train and test split
        train_n, test_n = int(len(image_datasets) * self.ratio), \
            len(image_datasets) - int(len(image_datasets)*self.ratio)
        logger.info(f'[loadData] train {train_n} test {test_n}')

        train, val = self._split(
            image_datasets, [train_n, test_n], self.save_mode, random=not self.ordered)

        self.train = train
        self.val = val
        self.train = MapDataset(self.train, self.tf['train'])
        self.val = MapDataset(self.val, self.tf['val'])

    def _split(self, dataset, lengths, save_mode, random):
        if sum(lengths) != len(dataset):
            raise ValueError(
                "Sum of input lengths does not equal the length of the input dataset!")
        if random:
            if save_mode == True:
                if os.path.isfile(self.dir_checkpoint+'random_indices.csv'):
                    indices = pd.read_csv(
                        self.dir_checkpoint+'random_indices.csv').iloc[:, 0].tolist()
                else:
                    indices = torch.randperm(sum(lengths)).tolist()
                    pd.DataFrame(indices).to_csv(self.dir_checkpoint +
                                                 'random_indices.csv', header=None, index=None)
            else:
                indices = torch.randperm(sum(lengths)).tolist()
        else:
            indices = list(range(sum(lengths)))
        return [Subset(dataset, indices[offset - length:offset]) for offset, length in
                zip(torch._utils._accumulate(lengths), lengths)]


class DatasetHelper:
    def __init__(self, dir_data, tf, ratio=0.8,
                 shuffle=True, batchsize=(32, 16), subclasses=[0, 1, 2, 3], binary=False, num_workers=4, save_mode=True, ordered=False):
        self.dir_checkpoint = './checkpoint/'
        self.tf = tf
        # train/test ratio
        self.ratio = ratio
        self.batchsize = batchsize
        self.subclasses = subclasses
        self.dir_data = dir_data
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.binary = binary
        self.save_mode = save_mode
        self.ordered = ordered
        # inital empty property
        self.class_names = None
        self.n_class = None
        self.dataset_sizes = None
        self.dataloader = None
        self._getTrainTestDataset()
        self._getDataLoader()

    def getDataLoader(self):
        return self.dataloader

    def getClassnames(self):
        return self.class_names

    def getNclass(self):
        return self.n_class

    def getDatasizes(self):
        return self.dataset_sizes

    def getBatchsize(self, batchtype: str = None) -> Union[int, Tuple[int, int]]:
        '''
        bacthtype: 'train', 'val', None\\
        return: int, int, (int,int)
        '''
        if batchtype == None:
            return self.batchsize
        elif batchtype == 'train':
            return self.batchsize[0]
        elif batchtype == 'val':
            return self.batchsize[1]

    def _getTrainTestDataset(self):
        image_datasets = datasets.ImageFolder(self.dir_data)
        if self.ordered == True:
            def sort_key(k): return int(k[0].split('_')[-1][0:-4])
            image_datasets.samples = sorted(
                image_datasets.samples, key=sort_key)
            image_datasets.imgs = image_datasets.samples
            self.image_datasets = image_datasets

        if self.subclasses != None:
            logger.debug(f'[getDataloader] subclasses {self.subclasses}')
            image_datasets = Subclass(
                image_datasets, self.subclasses, self.binary)

        # set class names
        self.class_names = image_datasets.classes

        # set number of classes
        self.n_class = len(self.class_names)

        # set train and test split
        train_n, test_n = int(len(image_datasets) * self.ratio), \
            len(image_datasets) - int(len(image_datasets)*self.ratio)
        logger.info(f'[loadData] train {train_n} test {test_n}')

        train, val = self._split(
            image_datasets, [train_n, test_n], self.save_mode, random=not self.ordered)

        self.train = train
        self.val = val

    def _getDataLoader(self):
        # apply data transformers
        train = MapDataset(self.train, self.tf['train'])
        val = MapDataset(self.val, self.tf['val'])

        # assemble datasets
        image_datasets = {'train': train, 'val': val}

        # apply data loader
        self.dataloader = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                          batch_size=bs,
                                                          shuffle=self.shuffle,
                                                          num_workers=self.num_workers)
                           for x, bs in [('train', self.batchsize[0]), ('val', self.batchsize[1])]}

        # set dataset sizes
        self.dataset_sizes = {x: len(image_datasets[x]) for x in [
            'train', 'val']}

    def _split(self, dataset, lengths, save_mode, random):
        if sum(lengths) != len(dataset):
            raise ValueError(
                "Sum of input lengths does not equal the length of the input dataset!")
        if random:
            if save_mode == True:
                if os.path.isfile(self.dir_checkpoint+'random_indices.csv'):
                    indices = pd.read_csv(
                        self.dir_checkpoint+'random_indices.csv').iloc[:, 0].tolist()
                else:
                    indices = torch.randperm(sum(lengths)).tolist()
                    pd.DataFrame(indices).to_csv(self.dir_checkpoint +
                                                 'random_indices.csv', header=None, index=None)
            else:
                indices = torch.randperm(sum(lengths)).tolist()
        else:
            indices = list(range(sum(lengths)))
        return [Subset(dataset, indices[offset - length:offset]) for offset, length in
                zip(torch._utils._accumulate(lengths), lengths)]


class DataMapBuilder():
    def __init__(self):
        self.dir_data = '../PlantVillage-Dataset/raw/color/'
        self.conf_fn = 'datamap.conf'
        self.conf_path = './checkpoint/'
        if not os.path.isdir(self.conf_path):
            os.mkdir(self.conf_path)

    def readDatamap(self):
        if not os.path.isfile(self.conf_path+self.conf_fn):
            self.buildDatamap()
        parser = et.XMLParser(remove_blank_text=True)
        tree = et.parse(self.conf_path+self.conf_fn, parser)
        root = tree.getroot()

        datamap = []
        for ic in root:
            name = ic.attrib['name']
            label = int(ic.attrib['label'])
            binary = ic.attrib['binary']
            binary_i = int(ic.attrib['binary_i'])
            count = int(ic.attrib['count'])
            start = int(ic.attrib['start'])
            end = int(ic.attrib['end'])
            datamap.append([name, label, binary, binary_i, count, start, end])

        datamap = pd.DataFrame(datamap, columns=[
                               'name', 'label', 'binary', 'binary_i', 'count', 'start', 'end'])

        return datamap

    def buildDatamap(self):
        self._readInfo()
        self._datamap_to_xml()

    def _datamap_to_xml(self):
        root = et.Element('datamap')
        self.tree = et.ElementTree(root)

        for name, binary, label, count, start, end in self.datamap:
            ic = et.SubElement(root, 'imageclass')
            ic.set('name', name)
            ic.set('label', str(label))
            ic.set('binary', binary)
            ic.set('binary_i', '0' if binary == 'healthy' else '1')
            ic.set('count', str(count))
            ic.set('start', str(start))
            ic.set('end', str(end))

        self.tree.write(self.conf_path+self.conf_fn, pretty_print=True)

    def _readInfo(self):
        dataset = ImageFolder(self.dir_data)
        self.binary_class = {}
        self.idx_to_class = {}
        for c, i in dataset.class_to_idx.items():
            self.idx_to_class[i] = c
            if c.find('healthy') != -1:
                self.binary_class[i] = 'healthy'
            else:
                self.binary_class[i] = 'unhealthy'

        datamap = []
        count = 0
        start = -1
        end = -1
        cur = -1
        last = -1
        total = 0
        for im, label in dataset:
            total += 1
            count += 1
            if label != last:
                last = cur
                cur = label
                end = total-2
                if last > -1:
                    name = self.idx_to_class[last]
                    binary = self.binary_class[last]
                    datamap.append([name, binary, last, count, start, end])
                    print([name, binary, last, count, start, end])
                last = label
                count = 0
                start = total-1

        self.datamap = datamap


class DataLoaderCV():
    def __init__(self, tf, binary, advimage_subfolder, kfold=5):
        self.pconf = PathConfig()
        self.binary = binary
        self.tf = tf
        self.dh_origin = DatasetHelper(self.pconf.dir_data, tf, ratio=0.8, shuffle=False, batchsize=[
            16, 1], binary=binary, save_mode=True)
        print(self.pconf.dir_advimages+advimage_subfolder)
        self.dh_attacked = DatasetHelper(self.pconf.dir_advimages + advimage_subfolder, tf,
                                         ratio=0, shuffle=False, batchsize=[16, 1], subclasses=None, binary=binary, save_mode=False, ordered=True)

        self.subclasses = settings.globals.subclasses
        data_origin = ImageFolder(self.pconf.dir_data)
        data_attacked = ImageFolder(
            self.pconf.dir_advimages+advimage_subfolder)

        data_attacked = self._sort_ImageFolder(data_attacked)

    def neg_pos_split(self):
        self.data_origin
        self.data_attacked = self._sort_ImageFolder(self.data_attacked)
        self.data_origin = Subclass(
            self.data_origin, self.subclasses, self.binary)

    def _sort_ImageFolder(self, dataset):
        def sort_key(k): return int(k[0].split('_')[-1][0:-4])

        dataset.samples = sorted(
            dataset.samples, key=sort_key)
        dataset.imgs = dataset.samples
        return dataset

    def split(self):
        count = 0
        for data, label, index in self.dh_origin.getDataLoader()['val']:
            count += 1

        print(count)
        count = 0
        for data, label, index in self.dh_attacked.getDataLoader()['val']:
            count += 1

        print(count)


def test():

    from torchvision import transforms
    tf = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ])
    }
    dh = DatasetHelper('../PlantVillage-Dataset/raw/color/', tf,
                       ratio=0.8, batchsize=(16, 100), subclasses=[0, 1, 2, 3], binary=False, num_workers=1)

    dl = dh.getDataLoader()
    count = 0

    dh.image_datasets
    for images, labels, _ in dl['val']:
        count += len(labels)

    logger.debug(f'[main] count {count}')


def test1():
    dmb = DataMapBuilder()
    datamap = dmb.readDatamap()
    print(datamap)


def test2():
    dh = DatasetHelper()
    from torchvision import transforms
    tf = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ])
    }
    dh(tf)
    print(dh.getDataloader())
    print(dh.getClassnames())
    print(dh.getDatasizes())


def test3():
    from torchvision import transforms
    datatransform = {
        'train': transforms.Compose([
            transforms.ToTensor()]),
        'val': transforms.Compose([
            transforms.ToTensor()]),
    }

    binary = False
    subfolder = f"vgg16_{'binary' if binary else 'multi'}_FGSM/eps_02/"
    dlcv = DataLoaderCV(datatransform, binary=binary,
                        advimage_subfolder=subfolder, kfold=5)

    dlcv.split()


if __name__ == '__main__':
    test()
