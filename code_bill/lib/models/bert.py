from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, List, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.file_utils import requires_datasets
from lib.utils import CleanData, TwitterData
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics.functional.classification import \
    confusion_matrix
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup
import torchvision.models as models
from lib.settings.config import settings

class BertMNLIFinetuner(pl.LightningModule):
    models.resnet101()
    def __init__(self,
                 pretrain_model_name,
                 learning_rate=2e-5,
                 adam_epsilon=1e-8,
                 warmup_steps=0,
                 weight_decay=0.0,
                 train_batch_size=32,
                 eval_batch_size=32,
                 layer_num=1,
                 freeze_type='all',
                 split_type='tvt',
                 **kwargs
                 ):
        super(BertMNLIFinetuner, self).__init__()
        self.save_hyperparameters()
        self.pretrain_model_name = pretrain_model_name
        self.split_type = split_type
        self.twdata = TwitterData(
            settings.data, self.pretrain_model_name, split_type=self.split_type)
        
        self.freeze_type = freeze_type
        self.layer_num = layer_num
        
        # use pretrained BERT
        self.bert = BertModel.from_pretrained(
            pretrain_model_name, output_attentions=True)
        # self.freeze_layer()
        # fine tuner (2 classes)
        self.num_classes = self.twdata.n_class
        
        
        self.classifier = self.make_classifier(self.layer_num)
        
        self.freeze_layer(freeze_type)

    def make_classifier(self, layer_num=1):
        layers = []
        sz = self.bert.config.hidden_size
        for l in range(layer_num-1):
            layers += [nn.Linear(sz, sz//2)]
            layers += [nn.ReLU(True)]
            layers += [nn.Dropout()]
            sz //= 2
        
        layers += [nn.Linear(sz, self.num_classes)]
        return nn.Sequential(*layers)

    def freeze_layer(self, freeze_type):
        if freeze_type == 'all':
            for param in self.bert.parameters():
                param.requires_grad = False
        elif freeze_type == 'no':
            for param in self.bert.parameters():
                param.requires_grad = True
        elif freeze_type == 'half':
            n = sum([1 for i in self.bert.parameters()])
            count = 0 
            for param in self.bert.parameters():
                param.requires_grad = False
                if count > n//2:
                    param.requires_grad = True
                count += 1

    def forward(self, input_ids, attention_mask, token_type_ids):

        h, _, attn = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)

        h_cls = h[:, 0]
        logits = self.classifier(h_cls)
        return logits, attn

    def prepare_data(self) -> None:
        self.twdata.prepare_data()

    def shared_my_step(self, batch, batch_nb, phase):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # fwd
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        # loss
        loss = F.cross_entropy(y_hat, label)

        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        acc = accuracy(y_hat, label)
        self.log(f'{phase}_loss', loss, prog_bar=True)
        self.log(f'{phase}_acc', acc, prog_bar=True)
        return {'loss': loss, f'{phase}_acc': acc, f'{phase}_label': label, f'{phase}_pred': y_hat}

    def epoch_end(self, outputs, phase):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        acc_mean = torch.stack([x[f'{phase}_acc'] for x in outputs]).mean()
        label = torch.cat([x[f'{phase}_label'] for x in outputs])
        
        pred = torch.cat([x[f'{phase}_pred'] for x in outputs])
        cm = confusion_matrix(pred, label)

        self.log(f'{phase}_loss_mean', loss_mean.cpu().detach().numpy())
        self.log(f'{phase}_acc_mean', acc_mean.cpu().detach().numpy())
        #print(cm)

    def training_step(self, batch, batch_nb):
        return self.shared_my_step(batch, batch_nb, 'train')

    def training_epoch_end(self, outputs) -> None:
        self.epoch_end(outputs, 'train')

    def validation_step(self, batch, batch_nb):
        return self.shared_my_step(batch, batch_nb, 'val')

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.epoch_end(outputs, 'val')

    def test_step(self, batch, batch_nb):
        return self.shared_my_step(batch, batch_nb, 'test')

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.epoch_end(outputs, 'test')

    def configure_optimizers(self):
        model = self.bert
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        print('*************************')
        print(self.hparams.learning_rate)
        print('*************************')
        optimizer = AdamW(model.parameters(
        ), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def setup(self, stage):
        self.twdata.setup()
        if stage == 'fit':

            self.total_steps = (
                100
            )

    def train_dataloader(self):
        return self.twdata.train_dataloader

    def test_dataloader(self):
        return self.twdata.test_dataloader

    def val_dataloader(self):
        return self.twdata.val_dataloader
