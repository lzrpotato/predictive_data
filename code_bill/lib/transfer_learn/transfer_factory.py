import os

import numpy as np
import pytorch_lightning as pl
from lib.models.bert import BertMNLIFinetuner
from lib.models.roberta import RoBERTaFinetuner
from lib.settings.config import settings
from lib.transfer_learn.param import Param, ParamGenerator
from lib.utils import Status
from lib.utils.twitter_data import TwitterData
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CometLogger
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

__all__ = ['TransferFactory']


class TransferFactory():
    def __init__(self):
        self.pg = ParamGenerator()
        exp = int(list(settings.transfer.param.exp)[0])
        self.status = Status(exp)
        
    def set_config(self, p: Param):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        pl.seed_everything(1234)
        monitor_key = 'acc'

        if monitor_key == 'loss':
            monitor_mode = 'min'
        elif monitor_key == 'acc':
            monitor_mode = 'max'
        else:
            raise ValueError(f'monitor_key "{monitor_key}" is incorrect')
        
        es_cb = EarlyStopping(
            monitor=f'val_{monitor_key}_epoch',
            patience=14,
            mode=monitor_mode,
        )
        model_name = p.pretrain_model.split('-')[0]
        ckp_cb = ModelCheckpoint(
            monitor=f'val_{monitor_key}_epoch',
            dirpath=settings.checkpoint,
            filename=model_name+'-best-model-{epoch:02d}-{val_acc_epoch:.3f}',
            save_top_k=1,
            mode=monitor_mode
        )
        self.ckp_cb = ckp_cb
        
        """
        comet_logger = CometLogger(
            api_key='RHywDLGIc61n40dBpkSqcmqp7',
            project_name='fakenews',  # Optional
            workspace='lzrpotato',
            experiment_name=p.experiment_name,
            offline=False,
        )
        """
        lr_monitor = LearningRateMonitor(
            logging_interval='epoch',
        )

        self.trainer = pl.Trainer(gpus=1,
                            max_epochs=50,
                            progress_bar_refresh_rate=0.1,
                            flush_logs_every_n_steps=100,
                            callbacks=[es_cb,ckp_cb,lr_monitor],
                            #logger=comet_logger
                            )

    def run(self):
        for p in self.pg.gen():
            print(p)
            if self.check_state(p):
                continue
            
            model_type ='DNNCV'
            if model_type == 'DNNCV':
                twdata = TwitterData(settings.data, p.pretrain_model,tree=p.tree,split_type=p.split_type,
                    max_tree_length=p.max_tree_len,limit=p.limit,cv=True,n_splits=5)
                twdata.setup_kfold()
                results_kfold = []
                for fold in twdata.kfold_gen():
                    self.set_config(p)
                    model = BertMNLIFinetuner(ep=p,fold=fold)
                    #model = RoBERTaFinetuner(ep=p,fold=fold)
                    if p.classifier.split('_')[0] == 'dense':
                        self.trainer.fit(model,train_dataloader=twdata.train_dataloader,val_dataloaders=twdata.val_dataloader)
                        test_result = self.trainer.test(test_dataloaders=twdata.test_dataloader)
                    elif p.classifier.split('_')[0] in ['svm','rf']:
                        if not os.path.isfile(f'./features/feamap_test_fd={fold}_{p.experiment_name}.npz'):
                            self.trainer.fit(model,train_dataloader=twdata.train_dataloader,val_dataloaders=twdata.val_dataloader)
                            test_result = self.trainer.test(model,test_dataloaders=[twdata.train_dataloader,twdata.test_dataloader])
                            
                        # load data
                        with open(f'./features/feamap_train_fd={fold}_{p.experiment_name}.npz','rb') as f:
                            dataset = np.load(f)
                        train_x, train_y = dataset[:,0:-1], dataset[:,-1]
                        
                        with open(f'./features/feamap_test_fd={fold}_{p.experiment_name}.npz','rb') as f:
                            test_dataset = np.load(f)
                        test_x, test_y = test_dataset[:,0:-1], test_dataset[:,-1]
                        
                        pca = PCA(0.98,svd_solver='full')
                        train_x = pca.fit_transform(train_x)
                        test_x = pca.transform(test_x)

                        ss = StandardScaler()
                        train_x = ss.fit_transform(train_x)
                        test_x = ss.transform(test_x)

                        # create svm model
                        svm_best_param = {'C': 10, 'gamma' : 0.001}
                        clf = OneVsRestClassifier(SVC(**svm_best_param),n_jobs=-1)

                        # fit
                        clf.fit(train_x, train_y)
                        train_score = clf.score(train_x, train_y)
                        test_score = clf.score(test_x,test_y)
                        
                        y_pred = clf.predict(test_x)
                        precision, recall, fscore, support = score(test_y,y_pred)
                        
                        # construct result dict
                        test_result = {'train_acc_epoch':train_score,
                                        'test_acc_epoch':test_score,
                                        'precision':precision,
                                        'recall':recall,
                                        'fscore':fscore,
                                        'support':support}

                    else:
                        test_result = []
                        print(f'warning {p.classifier} is not configured correctly')
                    if os.path.isfile(self.ckp_cb.best_model_path):
                        print('rm file', self.ckp_cb.best_model_path)
                        os.remove(self.ckp_cb.best_model_path)
                    else:
                        print(self.ckp_cb.best_model_path, ' file not exist')
                    results_kfold.append(test_result[0])
                self.save_state_kfold(p, results_kfold)
                #self.save_state(p, results_kfold[0])
            elif model_type == 'DNN':
                
                self.set_config(p)
                model = BertMNLIFinetuner(
                    p.pretrain_model,
                    layer_num=p.layer_num,
                    tree=p.tree,
                    max_tree_length=p.max_tree_len,
                    freeze_type=p.freeze_type,
                    split_type=p.split_type,
                    limit=p.limit,
                    dnn=p.dnn,
                    auxiliary=p.auxiliary,
                    ep=p,
                )
                model.prepare_data()
                self.trainer.fit(model)
                #print('load bm from checkpoint: ',self.ckp_cb.best_model_path)
                #model.load_from_checkpoint(self.ckp_cb.best_model_path)
                test_result = self.trainer.test(model)
                
                if os.path.isfile(self.ckp_cb.best_model_path):
                    print('rm file', self.ckp_cb.best_model_path)
                    os.remove(self.ckp_cb.best_model_path)
                else:
                    print(self.ckp_cb.best_model_path, ' file not exist')
                
                self.save_state(p, test_result[0])

            elif model_type == 'SVM':

                from sklearn.ensemble import RandomForestClassifier
                from sklearn.svm import SVC
                twdata = TwitterData(settings.data, p.pretrain_model,tree=p.tree,split_type=p.split_type,max_tree_length=p.max_tree_len)
                twdata.setup()
                train_x = np.array([np.concatenate((source.numpy(),tree.numpy()),axis=1) for source,_,_,tree,_ in twdata.dataset['train']])
                val_x = np.array([np.concatenate((source.numpy(),tree.numpy()),axis=1) for source,_,_,tree,_ in twdata.dataset['val']])
                train_y = np.array([label.numpy() for _,_,_,_,label in twdata.dataset['train']])
                val_y = np.array([label.numpy() for _,_,_,_,label in twdata.dataset['val']])
                
                clf = RandomForestClassifier()
                clf.fit(train_x,train_y)
                train_score = clf.score(train_x,train_y)
                val_score = clf.score(val_x,val_y)
                print(f'train_score {train_score}, val_score {val_score}')
    
    def save_state_kfold(self, p: Param, results: list):
        print(results)
        metrics = ['train_acc_epoch','val_acc_epoch','test_acc_epoch',
                    'precision','recall','fscore','support']
        ret = {}
        ret_avg = {}
        for m in metrics:
            if m not in results[0]:
                continue
            ret[m] = [r[m] for r in results]
            if type(np.mean(ret[m],axis=0)) == np.ndarray:
                ret_avg[m] = list(np.mean(ret[m],axis=0))
            else:
                ret_avg[m] = np.mean(ret[m],axis=0)
        
        ret_avg['ok'] = True
        print(ret_avg)

        self.status.save_state(p, ret_avg)

    def save_state(self, p: Param, result: dict):
        metrics = ['train_acc_epoch','val_acc_epoch','test_acc_epoch']
        ret = {}
        for metric in metrics:
            if metric not in result:
                continue
            ret[metric] = float(result[metric])

        self.status.save_state(p, ret)

    def check_state(self, p: Param):
        return self.status.check_state(p)
