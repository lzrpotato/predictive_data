from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.nn as nn
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
import pandas as pd

from lib.utils import TwitterData, CleanData


class BertMNLIFinetuner(pl.LightningModule):

    def __init__(self,
            pretrain_model_name,
            learning_rate = 2e-5,
            adam_epsilon = 1e-8,
            warmup_steps = 0,
            weight_decay = 0.0,
            train_batch_size = 32,
            eval_batch_size = 32,
            **kwargs
    ):
        super(BertMNLIFinetuner, self).__init__()
        self.save_hyperparameters()
        self.pretrain_model_name = pretrain_model_name
        self.twdata = TwitterData('../../rumor_detection_acl2017', self.pretrain_model_name)
        # use pretrained BERT
        self.bert = BertModel.from_pretrained(pretrain_model_name, output_attentions=True)
        
        # fine tuner (2 classes)
        self.num_classes = self.twdata.n_class
        self.W = nn.Linear(self.bert.config.hidden_size, self.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
      
        h, _, attn = self.bert(input_ids=input_ids, 
                         attention_mask=attention_mask, 
                         token_type_ids=token_type_ids)
        
        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn

    def prepare_data(self) -> None:
        self.twdata.prepare_data()

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch
        
        # fwd
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        # loss
        loss = F.cross_entropy(y_hat, label)
        
        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch
         
        # fwd
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        # loss
        loss = F.cross_entropy(y_hat, label)
        
        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}
    
    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())
        
        return {'test_acc': torch.tensor(test_acc)}

    def test_end(self, outputs):

        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
    
    def configure_optimizers(self):
        model = self.bert
        no_decay = ['bias','LayerNorm.weight']
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
        optimizer = AdamW(model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
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

