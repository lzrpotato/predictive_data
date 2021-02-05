import sys
sys.path.append('.')
from pyts.transformation import BOSS
import numpy as np
from lib.utils.twitter_data import TwitterData
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from lib.models.pt_cnn import PTCNN_C
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
import torch

def get_trainer(method, max_tree_len, fold):
    monitor_key = 'acc'
    if monitor_key == 'loss':
        monitor_mode = 'min'
    elif monitor_key == 'acc':
        monitor_mode = 'max'
    else:
        raise ValueError(f'monitor_key "{monitor_key}" is incorrect')
    if not os.path.isdir('./boss_checkpoint/'):
        os.mkdir('./boss_checkpoint/')

    es_cb = EarlyStopping(
        monitor=f'val_loss',
        patience=10,
        mode='min',
    )
    
    ckp_cb = ModelCheckpoint(
            monitor=f'val_{monitor_key}',
            dirpath='./boss_checkpoint/',
            filename= 'PTCNN_C' + '-sp=' + split_type + '-maxl=' + str(mll) + '-fold='+ str(fold) + '-{epoch:02d}-{val_acc:.3f}',
            save_top_k=1,
            mode=monitor_mode
    )
    comet_logger = CometLogger(
            api_key='RHywDLGIc61n40dBpkSqcmqp7',
            project_name='fakenews',  # Optional
            workspace='lzrpotato',
            experiment_name= f'BOSS-{method}-{max_tree_len}-{fold}',
            offline=False,
        )
    lr_monitor = LearningRateMonitor(
            logging_interval='epoch',
        )
    gpu_counts = torch.cuda.device_count()
    trainer = pl.Trainer(
        gpus=gpu_counts,
        max_epochs=500,
        progress_bar_refresh_rate=0,
        flush_logs_every_n_steps=100,
        callbacks=[ckp_cb,lr_monitor],
        logger=comet_logger,
    )
    return trainer, ckp_cb

split_type = '15_tv'
fold = 0
results_mll = {}
for mll in range(100,1001,100):
    if mll not in [800,900,1000]:
        #continue
        continue
    td = TwitterData(tree='tree',max_tree_length=mll,datatype='all',split_type=split_type,subclass=False,cv=True,kfold_deterministic=True)
    td.setup()
    nclass = td.n_class
    results_folds = {}
    results_mll[mll] = results_folds
    for fold in range(5):
        td.kfold_get_by_fold(fold)
        for method in ['CNNTK_64', 'RAW']:
            if method not in results_folds:
                results_folds[method] = []
            # boss = BOSS(word_size=2,n_bins=4,window_size=12,sparse=False)
            if method == 'RAW':
                X_train, y_train = td.train_data[1], td.train_data[2]
                X_test, y_test = td.test_data[1], td.test_data[2]
                X_train = X_train.reshape(X_train.shape[0],-1)
                X_test = X_test.reshape(X_test.shape[0],-1)
                print(X_train.shape)
                clf_ = OneVsRestClassifier(SVC()).fit(X_train, y_train)
                score_train_raw = clf_.score(X_train, y_train)
                score_test_raw = clf_.score(X_test, y_test)
                results_folds['RAW'].append(score_test_raw)
                print('raw ',mll, score_train_raw, score_test_raw)

                '''
                X_train_boss = boss.fit_transform(td.train_data[1])
                X_test_boss = boss.transform(td.test_data[1])
                print(X_train_boss.shape)
                clf = OneVsRestClassifier(SVC()).fit(X_train_boss,td.train_data[2])
                score_train = clf.score(X_train_boss,td.train_data[2])
                score_test = clf.score(X_test_boss,td.test_data[2])

                print('boss ',mll,score_train,score_test)
                '''
            else:
                trainer, ckp_cb = get_trainer(method, mll, fold)
            
                dnn, fst_c = method.split('_')
                model = PTCNN_C(td.feature_dim,mll,nclass,int(fst_c),pool='adaptive',method=dnn)

                trainer.fit(model,train_dataloader=td.train_dataloader,val_dataloaders=td.val_dataloader)
                results = trainer.test(test_dataloaders=td.test_dataloader)
                results_folds[method].append(results[0]['test_acc'])
                print(f'{method} results  ',results)
                if os.path.isfile(ckp_cb.best_model_path):
                    print('rm',ckp_cb.best_model_path)
                    os.remove(ckp_cb.best_model_path)
                else:
                    print('cannot find file')
    for k, v in results_folds.items():
        if v == []:
            pass
        print(f'mll {mll} method {k} acc {np.mean(v)}')
    
for k, folds in results_mll.items():
    for m, f in folds.items():
        if f == []:
            pass
        print(f'mll {k} method {m} acc {np.mean(f)}')
