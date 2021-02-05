import os
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.settings.config import settings
from lib.transfer_learn.param import Param
from lib.utils.twitter_data import TwitterData
from pytorch_lightning.metrics import Accuracy
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.sgd import SGD
from transformers import (AdamW, RobertaModel, get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)
from lib.models.pt_cnn import PTCNN, CNN, CNN_, CNNOri, CNN_AVG, CNN_TKF

import logging
logger = logging.getLogger('pytorch_lightning.roberta')

__all__ = ['BertMNLIFinetuner']


class RoBERTaFinetuner(pl.LightningModule):
    def __init__(self,
                 ep: Param,
                 fold=None,
                 feature_dim=None,
                 max_epoch=50,
                 learning_rate=2e-5,
                 adam_epsilon=1e-8,
                 warmup_steps=0,
                 weight_decay=0.0,
                 train_batch_size=32,
                 eval_batch_size=32,
                 **kwargs
                 ):
        super(RoBERTaFinetuner, self).__init__()
        self.save_hyperparameters()
        self.fold = fold
        self.ep = ep
        if ep.pretrain_model.split('-')[0] != 'roberta':
            raise ValueError(f'[Error] pretrain model name is wrong {ep.pretrain_model}')
        self.pretrain_model_name = ep.pretrain_model
        self.split_type = ep.split_type
        self.tree = ep.tree
        self.max_tree_length = ep.max_tree_len
        self.limit = ep.limit
        self.dnn = ep.dnn
        self.auxiliary = ep.auxiliary
        self.max_epoch = max_epoch
        
        self.twdata = TwitterData(
            settings.data, self.pretrain_model_name,tree=ep.tree,split_type=ep.split_type,max_tree_length=ep.max_tree_len,limit=ep.limit)
        
        self.freeze_type = ep.freeze_type
        self.classifier_type = ep.classifier
        if self.classifier_type.split('_')[0] == 'dense':
            self.layer_num = int(self.classifier_type.split('_')[1])
        elif self.classifier_type.split('_')[0] in ['svm','rf']:
            self.layer_num = 1
        self.reduction = ep.reduction
        if feature_dim:
            self.feature_dim = feature_dim
        else:
            self.feature_dim = self.twdata.featue_dim
        self.num_classes = self.twdata.n_class

        self._create_model()
        self.freeze_layer(self.freeze_type)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
    
    def _create_model(self):
        # use pretrained BERT
        self.roberta = RobertaModel.from_pretrained(
            self.pretrain_model_name, output_attentions=True)

        self.tree_hidden_dim = 0
        tree_nn_type = self.dnn
        if self.tree != 'none':
            if tree_nn_type == 'MLP':
                self.tree_layer = nn.Sequential(
                                nn.BatchNorm1d(self.feature_dim*self.max_tree_length),
                                nn.Linear(self.feature_dim*self.max_tree_length,self.feature_dim*self.max_tree_length//2),
                                nn.ReLU(True),
                                nn.BatchNorm1d(self.feature_dim*self.max_tree_length//2),
                                nn.Linear(self.feature_dim*self.max_tree_length//2,self.feature_dim*self.max_tree_length//2),
                                nn.ReLU(True),
                                nn.BatchNorm1d(self.feature_dim*self.max_tree_length//2),
                                nn.Linear(self.feature_dim*self.max_tree_length//2,self.feature_dim*self.max_tree_length//4),
                                nn.ReLU(True),
                                nn.Dropout(),
                            )
                self.tree_hidden_dim = self.feature_dim*self.max_tree_length//4
            elif tree_nn_type == 'CNN':
                self.tree_layer = nn.Sequential(
                                nn.Unflatten(1, (1,self.feature_dim*self.max_tree_length)),                # b,1,self.feature_dim*self.max_tree_length
                                nn.BatchNorm1d(1),                       
                                nn.Conv1d(1,8,kernel_size=3,stride=2),  # b,8,self.feature_dim*self.max_tree_length//2
                                nn.AvgPool1d(3,2),                      # b,8,self.feature_dim*self.max_tree_length//4
                                nn.ReLU(True),
                                nn.BatchNorm1d(8),
                                nn.Conv1d(8,16,kernel_size=3,stride=2), # b,16,self.feature_dim*self.max_tree_length//8
                                nn.ReLU(True),
                                nn.Flatten(1,-1),
                            )
                self.tree_hidden_dim = (self.feature_dim*self.max_tree_length//8-1)*16
            
            elif tree_nn_type == 'CNN_test':
                self.tree_layer = nn.Sequential(
                                nn.Unflatten(1, (1,self.feature_dim*self.max_tree_length)),                # b,1,self.feature_dim*self.max_tree_length
                                nn.BatchNorm1d(1),                       
                                nn.Conv1d(1,8,kernel_size=3,stride=2),  # b,8,self.feature_dim*self.max_tree_length//2
                                nn.AvgPool1d(3,2),                      # b,8,self.feature_dim*self.max_tree_length//4
                                nn.BatchNorm1d(8),
                                nn.ReLU(True),
                                nn.Conv1d(8,16,kernel_size=3,stride=2), # b,16,self.feature_dim*self.max_tree_length//8
                                nn.BatchNorm1d(16),
                                nn.ReLU(True),
                                nn.Flatten(1,-1),
                            )

                self.tree_hidden_dim = (self.feature_dim*self.max_tree_length//8-1)*16
            elif tree_nn_type == 'CNN_test2':
                self.tree_layer = CNN(self.feature_dim, self.max_tree_length)
                self.tree_hidden_dim = self.tree_layer.out_dim
            elif tree_nn_type == 'CNN_test3':
                self.tree_layer = CNN_(self.feature_dim, self.max_tree_length)
                self.tree_hidden_dim = self.tree_layer.out_dim
            elif tree_nn_type == 'CNNOri':
                self.tree_layer = CNNOri(self.feature_dim,self.max_tree_length,fst_p=16)
                self.tree_hidden_dim = self.tree_layer.out_dim
            elif tree_nn_type.split('_')[0] == 'CNNAVG':
                fst_p = int(tree_nn_type.split('_')[1])
                self.tree_layer = CNN_AVG(self.feature_dim,self.max_tree_length,fst_p=fst_p)
                self.tree_hidden_dim = self.tree_layer.out_dim
            elif tree_nn_type.split('_')[0] == 'CNNRes':
                fst_c = int(tree_nn_type.split('_')[1])
                self.tree_layer = PTCNN(self.feature_dim,self.max_tree_length,fst_p=fst_c,blocks=2,pool='adaptive')
                self.tree_hidden_dim = self.tree_layer.out_dim
            elif tree_nn_type.split('_')[0] == 'CNNTKF':
                fst_c = int(tree_nn_type.split('_')[1])
                self.tree_layer = CNN_TKF(self.feature_dim,self.max_tree_length,fst_p=fst_c,dropout=False)
                self.tree_hidden_dim = self.tree_layer.out_dim
            elif tree_nn_type.split('_')[0] == 'PTCNN':
                fst_c = int(tree_nn_type.split('_')[1])
                self.tree_layer = PTCNN(self.feature_dim, self.max_tree_length, fst_c)
                self.tree_hidden_dim = self.tree_layer.out_dim
            
            elif tree_nn_type == 'LSTM':
                self.tree_hidden_dim = 100
                self.lstm1_num_layers = 3
                self.tree_layer = nn.Sequential(
                                nn.Unflatten(1,(self.max_tree_length,self.feature_dim)),
                                nn.LSTM(self.feature_dim, self.tree_hidden_dim,num_layers=self.lstm1_num_layers,batch_first=True),
                            )

            self.classifier1 = self.make_classifier(self.tree_hidden_dim,self.layer_num)
            # self.classifier1 = self.make_classifier(self.tree_hidden_dim,self.layer_num)

        self.classifier = self.make_classifier(self.roberta.config.hidden_size+self.tree_hidden_dim,self.layer_num)

    def make_classifier(self, hidden_size, layer_num=1):
        layers = []
        sz = hidden_size
        #layers += [nn.BatchNorm1d(sz)]
        for l in range(layer_num-1):
            layers += [nn.Linear(sz, sz//2)]
            layers += [nn.ReLU(True)]
            layers += [nn.Dropout()]
            sz //= 2
        
        #layers += [nn.Dropout(0.2)]
        layers += [nn.Linear(sz, self.num_classes)]
        return nn.Sequential(*layers)

    def freeze_layer(self, freeze_type):
        if freeze_type == 'all':
            for param in self.roberta.parameters():
                param.requires_grad = False
        elif freeze_type == 'no':
            for param in self.roberta.parameters():
                param.requires_grad = True
        elif freeze_type == 'half':
            n = sum([1 for i in self.roberta.parameters()])
            count = 0 
            for param in self.roberta.parameters():
                param.requires_grad = False
                if count > n//2:
                    param.requires_grad = True
                count += 1

    def forward(self, input_ids, attention_mask, tree, phase):
        return_dict = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               return_dict=True)
        h = return_dict['last_hidden_state']
        h_cls = h[:, 0]

        # input of lstm (seq_len, batch, feature_dim)
        # output of lstm (seq_len, batch, num_directions*hidden_size)
        cls_out = None
        if self.tree != 'none':
            if self.dnn == 'LSTM':
                lstmout, (hn, cn) = self.tree_layer(tree)
                # get the last hidden state (batch, hidden_dim)
                tree_out = hn[-1]
            else:
                tree_out = self.tree_layer(tree)
            
            # concate bert and lstm output
            cls_out = torch.cat((h_cls,tree_out),dim=1)
            if phase=='test' and self.ep.classifier.split('_')[0] in ['svm','rf']:
                return cls_out.view(cls_out.size(0),-1), tree_out.view(tree_out.size(0),-1)
            logits_1 = self.classifier1(tree_out.view(tree_out.size(0),-1))
            logits = self.classifier(cls_out.view(cls_out.size(0),-1))
            return logits, logits_1
        else:
            cls_out = h_cls
            if phase=='test' and self.ep.classifier.split('_')[0] in ['svm','rf']:
                return cls_out.view(cls_out.size(0),-1)
            logits = self.classifier(cls_out.view(cls_out.size(0),-1))
        
            return logits

    def prepare_data(self) -> None:
        logger.debug('prepare_data called')
        #self.twdata.prepare_data()
        #self.twdata.setup()

    def shared_my_step(self, batch, batch_nb, phase):
        # batch
        if self.tree != 'none':
            input_ids, attention_mask, tree, label = batch
            # fwd
            y_hat, logits_1 = self.forward(input_ids, attention_mask,tree, phase)
            loss_1 = F.cross_entropy(y_hat, label)
            loss_2 = F.cross_entropy(logits_1,label)
            if phase == 'train' and self.auxiliary:
                loss = loss_1 + 0.4*loss_2
                
            else:
                loss = loss_1
                
        else:
            input_ids, attention_mask, label = batch
            # fwd
            y_hat = self.forward(input_ids, attention_mask, None, phase)
            loss = F.cross_entropy(y_hat, label)

        # acc
        a, y_hat = torch.max(y_hat, dim=1)

        self.log(f'{phase}_loss_step', loss, sync_dist=True, prog_bar=True)
        self.log(f'{phase}_loss_epoch', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'loss': loss, f'{phase}_label': label, f'{phase}_pred': y_hat}

    def epoch_end(self, outputs, phase):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        
        label = torch.cat([x[f'{phase}_label'] for x in outputs]).cpu().detach().numpy()
        pred = torch.cat([x[f'{phase}_pred'] for x in outputs]).cpu().detach().numpy()

        if phase == 'test':
            #fig, ax = plt.subplots(figsize=(16, 12))
            #plot_confusion_matrix(label, pred, ax=ax)
            #self.logger.experiment.log_image(settings.fig+f'test_cm.png')
            #self.logger.experiment.log_confusion_matrix(label,pred,file_name=f'{phase}_confusion_matrix.json',max_example_per_cell=self.num_classes)
            precision, recall, fscore, support = score(label,pred)
            return precision, recall, fscore, support

    def training_step(self, batch, batch_nb):
        phase = 'train'
        outputs = self.shared_my_step(batch, batch_nb, phase)
        self.log(f'{phase}_acc_step', self.train_acc(outputs[f'{phase}_label'],outputs[f'{phase}_pred']), sync_dist=True, prog_bar=True)
        return outputs

    def training_epoch_end(self, outputs) -> None:
        phase = 'train'
        self.log(f'{phase}_acc_epoch', self.train_acc.compute())
        self.epoch_end(outputs, phase)

    def validation_step(self, batch, batch_nb):
        phase = 'val'
        outputs = self.shared_my_step(batch, batch_nb, phase)
        self.log(f'{phase}_acc_step', self.val_acc(outputs[f'{phase}_label'],outputs[f'{phase}_pred']), sync_dist=True, prog_bar=True)
        return outputs

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        phase = 'val'
        self.log(f'{phase}_acc_epoch', self.val_acc.compute())
        print(f'\nval_acc_epoch {self.val_acc.compute()}')
        self.epoch_end(outputs, phase)

    def test_step(self, batch, batch_nb, dataloader_idx=-1):
        phase = 'test'
        #outputs = self.shared_my_step(batch, batch_nb, phase)
        
        if dataloader_idx == -1:
            self.multi_dl = False
        else:
            self.multi_dl = True

        if self.classifier_type.split('_')[0] == 'dense':
            if self.tree != 'none':
                input_ids, attention_mask, tree, label = batch
                # fwd
                y_hat, logits_1 = self.forward(input_ids, attention_mask,tree,phase)
                loss = F.cross_entropy(y_hat, label)
                
            else:
                input_ids, attention_mask, label = batch
                # fwd
                y_hat = self.forward(input_ids, attention_mask, None,phase)
                loss = F.cross_entropy(y_hat, label)
             # acc
            a, y_hat = torch.max(y_hat, dim=1)

            self.log(f'{phase}_loss_step', loss, sync_dist=True, prog_bar=True)
            self.log(f'{phase}_loss_epoch', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{phase}_acc_step', self.test_acc(label, y_hat), sync_dist=True, prog_bar=True)
            return {'loss': loss, f'{phase}_label': label, f'{phase}_pred': y_hat}
        else:
            if self.tree != 'none':
                input_ids, attention_mask, tree, label = batch
                y_fea_map, y_fea_map_tree = self.forward(input_ids, attention_mask,tree, phase)
            else:
                input_ids, attention_mask, label = batch
                # fwd
                y_fea_map = self.forward(input_ids, attention_mask, None, phase)
            return {'fea_map':y_fea_map, 'labels': label}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        phase = 'test'
        if self.classifier_type.split('_')[0] == 'dense':
            self.log(f'{phase}_acc_epoch', self.test_acc.compute())
            precision, recall, fscore, support = self.epoch_end(outputs, phase)
            return {'precision':precision, 'recall':recall, 'fscore':fscore, 'support':support}
        else:
            if self.multi_dl:
                for i, output in enumerate(outputs):
                    fea_map = torch.cat([x['fea_map'] for x in output]).cpu().detach().numpy()
                    labels = torch.cat([x['labels'] for x in output]).cpu().detach().numpy()
                    n_dataset = np.concatenate((fea_map,labels.reshape(-1,1)), axis=1)
                    
                    if i == 0:
                        fn = f'feamap_train_fd={self.fold}_' + self.ep.experiment_name + '.npz'
                    else:
                        fn = f'feamap_test_fd={self.fold}_' + self.ep.experiment_name + '.npz'
                    path = os.path.join('./features/',fn)
                    with open(path, 'wb') as f:
                        np.save(f, n_dataset)
                        logger.info(f'save fea_map to {path}')
            else:
                fea_map = torch.cat([x['fea_map'] for x in outputs]).cpu().detach().numpy()
                labels = torch.cat([x['labels'] for x in outputs]).cpu().detach().numpy()
                n_dataset = np.concatenate((fea_map,labels.reshape(-1,1)), axis=1)
                
                fn = f'feamap_fd={self.fold}_' + self.ep.experiment_name + '.npz'
                path = os.path.join('./features/',fn)
                with open(path, 'wb') as f:
                    np.save(f, n_dataset)
                    logger.info(f'save fea_map to {path}')

    def configure_optimizers(self):
        if self.tree != 'none':
            optimizer1 = AdamW([
                        {'params': self.roberta.parameters(), 'lr': 2e-5},
                        {'params': self.classifier.parameters(), 'lr':1e-3,'weight_decay':1e-2},
                        {'params': self.classifier1.parameters(), 'lr':1e-3,'weight_decay':1e-2},
                        {'params': self.tree_layer.parameters(), 'lr': 1e-3,'weight_decay':1e-2}
                    ],
                eps=self.hparams.adam_epsilon)
        else:
            optimizer1 = AdamW([
                        {'params': self.roberta.parameters(), 'lr': 2e-5},
                        {'params': self.classifier.parameters(), 'lr':1e-3,'weight_decay':1e-2},
                    ],
                lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler_cosine = get_linear_schedule_with_warmup(optimizer1
                ,num_warmup_steps=self.max_epoch//10, num_training_steps=self.max_epoch)
        #scheduler1 = StepLR(optimizer=optimizer1, step_size=7, gamma=0.1)
        scheduler = {
            'scheduler': scheduler_cosine,
            'name': 'lr_scheduler_1',
        }
        return [optimizer1], [scheduler]

    def _configure_optimizers(self):
        model = self.roberta
        optimizer1 = AdamW(model.parameters(), 
            lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        optimizer2 = AdamW(self.tree_layer.parameters(), lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon)
        
        optimizer3 = AdamW(self.classifier.parameters(), lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon)

        optimizers = [optimizer1, optimizer2, optimizer3]

        scheduler1 = StepLR(optimizer=optimizer1, step_size=5, gamma=0.01)
        scheduler2 = StepLR(optimizer=optimizer2, step_size=10, gamma=0.1)
        scheduler3 = StepLR(optimizer=optimizer3, step_size=10, gamma=0.1)

        schedulers = [
            {
                'scheduler': scheduler1,
                'name': 'lr_scheduler_1',
            },
            {
                'scheduler': scheduler2,
                'name': 'lr_scheduler_2',
            },
            {
                'scheduler': scheduler3,
                'name': 'lr_scheduler_3',
            },
        ]
        return optimizers, schedulers

    def setup(self, stage):
        logger.debug('setup called')
        pass

    def train_dataloader(self):
        return self.twdata.train_dataloader

    def test_dataloader(self):
        return self.twdata.test_dataloader

    def val_dataloader(self):
        return self.twdata.val_dataloader
