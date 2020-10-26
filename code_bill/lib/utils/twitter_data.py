from typing import List, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class TwitterData(pl.LightningDataModule):
    def __init__(
        self,
        rootpath,
        pretrain_tokenizer_model,
        max_seq_length=128,
        train_batch_size=32,
        val_batch_size=32,
        test_batch_size=32,
        split_type='tvt',
        **kwargs
    ):
        super().__init__()
        self.rootpath = rootpath
        self.pretrain_tokenizer_model = pretrain_tokenizer_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrain_tokenizer_model, use_fast=True)
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.n_class = 4
        self.split_type = split_type
        if self.split_type not in ['1516','tv','tvt','15_tvt','16_tvt']:
            print('warning: split_type invalid!')
            self.split_type = '641620'

    def setup(self):
        self.train, self.val, self.test = None, None, None
        self._load_data()
        self._data_split(self.split_type)
        self.dataset = {'train': self.train, 'val': self.val, 'test': self.test}
        for split in self.dataset.keys():
            d = self.dataset[split]
            if d is None:
                continue
            data = pd.DataFrame(d)
            data.columns = ['data', 'labels']
            self.dataset[split] = self.convert_to_features(data)

        self._set_dataloader()

    def prepare_data(self):
        AutoTokenizer.from_pretrained(
            self.pretrain_tokenizer_model, use_fast=True)

    def convert_to_features(self, example, indices=None):
        features = self.tokenizer.batch_encode_plus(
            list(example['data']),
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
        )

        features_ = []

        for i, l in enumerate(example['labels']):
            features_.append((torch.tensor(features['input_ids'][i]), torch.tensor(features['token_type_ids'][i]),
                              torch.tensor(features['attention_mask'][i]), example['labels'][i]))
        return features_

    def _find_class(self, label1, label2):
        label = np.concatenate((label1, label2))
        classes = sorted(np.unique(label))
        self.class_to_index = {classname: i for i,
                               classname in enumerate(classes)}
        self.class_names = classes
        self.n_class = len(classes)
        self.classes = [i for i in range(len(self.class_names))]

    def _class_to_index(self, label):
        index = np.vectorize(self.class_to_index.__getitem__)(label)
        return index

    def _load_data(self):
        tw15source = self.rootpath + '/twitter15/source_tweets.txt'
        tw16source = self.rootpath + '/twitter16/source_tweets.txt'
        tw15label = self.rootpath + '/twitter15/label.txt'
        tw16label = self.rootpath + '/twitter16/label.txt'

        tw15_text = self._read_text(tw15source)

        tw16_text = self._read_text(tw16source)

        tw15_label = self._read_label(tw15label)

        tw16_label = self._read_label(tw16label)

        self.tw15_X, self.tw15_y = self._combine_text_label(
            tw15_text, tw15_label)
        self.tw16_X, self.tw16_y = self._combine_text_label(
            tw16_text, tw16_label)

        self._find_class(self.tw15_y, self.tw16_y)

        self.tw15_y = self._class_to_index(self.tw15_y)
        self.tw16_y = self._class_to_index(self.tw16_y)

    def _data_split(self, split_type):
        if split_type == '1516':
            self.train = [[data, label]
                          for data, label in zip(self.tw15_X, self.tw15_y)]
            self.test = [[data, label]
                         for data, label in zip(self.tw16_X, self.tw16_y)]
        if split_type == 'tt':
            X = np.concatenate((self.tw15_X, self.tw16_X))
            y = np.concatenate((self.tw15_y, self.tw16_y))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=1, stratify=self.self.classes)

            self.train = [[data, label]
                          for data, label in zip(X_train, y_train)]
            self.test = [[data, label] for data, label in zip(X_test, y_test)]
        if split_type == 'tvt':
            X = np.concatenate((self.tw15_X, self.tw16_X))
            y = np.concatenate((self.tw15_y, self.tw16_y))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, random_state=1, shuffle=True, stratify=y)

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, train_size=0.8, random_state=1, shuffle=True, stratify=y_train)

            
            self.train = [[data, label] for data, label in zip(X_train, y_train)]
            self.val = [[data, label] for data, label in zip(X_val, y_val)]
            self.test = [[data, label] for data, label in zip(X_test, y_test)]
        if split_type == '15_tvt':
            X_train, X_test, y_train, y_test = train_test_split(
                self.tw15_X, self.tw15_y, train_size=0.8, random_state=1, shuffle=True, stratify= self.tw15_y)

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, train_size=0.8, random_state=1, shuffle=True, stratify=y_train)

            self.train = [[data, label] for data, label in zip(X_train, y_train)]
            self.val = [[data, label] for data, label in zip(X_val, y_val)]
            self.test = [[data, label] for data, label in zip(X_test, y_test)]

        if split_type == '16_tvt':
            X_train, X_test, y_train, y_test = train_test_split(
                self.tw16_X, self.tw16_y, train_size=0.8, random_state=1, shuffle=True, stratify=self.tw16_y)

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, train_size=0.8, random_state=1, shuffle=True, stratify=y_train)

            self.train = [[data, label] for data, label in zip(X_train, y_train)]
            self.val = [[data, label] for data, label in zip(X_val, y_val)]
            self.test = [[data, label] for data, label in zip(X_test, y_test)]

    def _set_dataloader(self, shuffle=True):
        
        self._train_data = DataLoader(self.dataset['train'],
                                      batch_size=self.train_batch_size,
                                      shuffle=shuffle,
                                      num_workers=4)
        self._test_data = DataLoader(self.dataset['test'],
                                     batch_size=self.test_batch_size,
                                     shuffle=False,
                                     num_workers=4)
        if self.dataset['val'] is not None:
            self._val_data = DataLoader(self.dataset['val'],
                                        batch_size=self.val_batch_size,
                                        shuffle=False,
                                        num_workers=4)

    @property
    def train_dataloader(self) -> DataLoader:
        return self._train_data

    @property
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._test_data

    @property
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._val_data

    def _combine_text_label(self, texts, labels):
        text_label = []
        for id, text in texts.items():
            label = labels[id]
            text_label.append([text, label])

        text_label = np.array(text_label)

        return text_label[:, 0], text_label[:, 1]

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

    def _read_label(self, path):
        pairs = {}
        with open(path, mode='r') as f:
            for line in f:
                label, id = line.split(':')
                if id not in pairs.keys():

                    pairs[int(id)] = label
                else:
                    print('error')
        return pairs
