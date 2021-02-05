import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import math
from transformers import (AdamW, BertModel, get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)


class BasicBlock2d(nn.Module):
    def __init__(
        self, 
        inplanes: int, 
        planes: int, 
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

class PTCNN2d(nn.Module):
    def __init__(self, feature_d, max_tree_len, fst_c, blocks=1, pool='adaptive'):
        super(PTCNN2d, self).__init__()
        inplanes = feature_d
        self.input_layer = nn.Sequential(
            #nn.Unflatten(1, (inplanes,max_tree_len)),
            nn.BatchNorm2d(inplanes)
        )
        
        nb = np.sqrt(max_tree_len) // 8
        real_nb = 1 if nb <= 1 else int(nb)
        blocks = 1
        print(f'#### blocks of cnn ({real_nb*blocks}) ####')
        layers = []
        planes = fst_c
        layers += self._make_layer(BasicBlock2d,inplanes,planes,blocks=blocks,stride=2)
        inplanes = planes

        for bi in range(1,int(nb)):
            planes = fst_c * 2 ** bi
            layers += self._make_layer(BasicBlock2d,inplanes,planes,blocks=blocks,stride=2)
            inplanes = planes
        
        self.layers = nn.Sequential(*layers)
        
        if pool not in ['adaptive','global']:
            raise ValueError(f'pool is incorrect {pool}')

        self.pool = pool
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        if pool == 'adaptive':
            self.out_dim = planes
        elif pool == 'global':
            self.out_dim = math.ceil(max_tree_len / (2**(real_nb)))

    def get_out_dim(self):
        if self.out_dim is None:
            raise UnboundLocalError('unbound self.out_dim')

        return self.out_dim

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1),
                nn.BatchNorm2d(planes)
            )

        layers = []
        layers += [block(inplanes,planes,stride, downsample)]
        
        inplanes = planes
        for _ in range(1, blocks):
            layers += [BasicBlock1d(inplanes, planes)]
        
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.layers(x)
        print('layers shape', x.shape)
        if self.pool == 'global':
            x = torch.mean(x,1)
            print('mean shape', x.shape)
            x = torch.flatten(x,1)
            print('flatten shape', x.shape)
        else:
            x = self.avgpool(x)
            print('avgpool shape', x.shape)
            x = torch.flatten(x,1)
            print('flatten shape', x.shape)

        return x

class BasicBlock1d(nn.Module):
    def __init__(
        self, 
        inplanes: int, 
        planes: int, 
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes,planes,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

class PTCNN(nn.Module):
    def __init__(self, feature_d, max_tree_len, fst_p=16, blocks=1, pool='global'):
        super(PTCNN, self).__init__()

        nb = 0
        i = max_tree_len/8
        while i/4 > 1:
            i /= 4
            nb += 1
        nb+=1

        inplanes = feature_d
        planes = fst_p

        print(f'#### blocks of cnn ({nb*blocks}) ####')
        layers = []
        for bi in range(0,int(nb)):
            layers += self._make_layer(BasicBlock1d,inplanes,planes,blocks=blocks,stride=2)
            inplanes = planes
            planes = fst_p * 2
        
        self.layers = nn.Sequential(*layers)
        
        if pool not in ['adaptive','global']:
            raise ValueError(f'pool is incorrect {pool}')

        self.pool = pool
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        if pool == 'adaptive':
            self.out_dim = planes
        elif pool == 'global':
            self.out_dim = math.ceil(max_tree_len / (2**(nb)))

        print(f'PTCNN planes {planes} max_tree_len {max_tree_len} nb {nb}')

    def get_out_dim(self):
        if self.out_dim is None:
            raise UnboundLocalError('unbound self.out_dim')

        return self.out_dim

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes,planes,kernel_size=3,stride=stride,padding=1),
                nn.BatchNorm1d(planes)
            )

        layers = []
        layers += [block(inplanes,planes,stride, downsample)]
        
        inplanes = planes
        for _ in range(1, blocks):
            layers += [BasicBlock1d(inplanes, planes)]
        
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        #x = self.input_layer(x)
        x = self.layers(x)
        #print('layers shape', x.shape)
        if self.pool == 'global':
            x = torch.mean(x,1)
            #print('mean shape', x.shape)
        else:
            x = self.avgpool(x)
            #print('avgpool shape', x.shape)
            x = torch.flatten(x,1)
            #print('flatten shape', x.shape)

        return x

class CNN_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel, stride, padding=0, pool=True):
        super(CNN_BasicBlock,self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(True)
        self.pool = pool
        self.factor = 2
        if self.pool:
            self.avgpool = nn.AvgPool1d(kernel_size=kernel, stride=stride, padding=padding)
            self.factor = 4
        

    def forward(self, x):
        #print(f'cnn basic input {x.shape}')
        x = self.conv1(x)
        #print(f'conv1 input {x.shape}')
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.avgpool(x)
            #print(f'avgpool input {x.shape}')
        #print(f'cnn basic output {x.shape}')
        return x

class CNNOri(nn.Module):
    def __init__(self,feature_d, max_tree_len, fst_p=8):
        super().__init__()
        inplanes = feature_d
        self.tree_layer = nn.Sequential(
            #nn.Unflatten(1, (feature_d,max_tree_len)),                # b,1,self.feature_dim*self.max_tree_length
            nn.BatchNorm1d(inplanes),                       
            nn.Conv1d(inplanes,fst_p,kernel_size=3,stride=2),  # b,8,self.feature_dim*self.max_tree_length//2
            nn.AvgPool1d(3,2),                      # b,8,self.feature_dim*self.max_tree_length//4
            nn.ReLU(True),
            nn.BatchNorm1d(fst_p),
            nn.Conv1d(fst_p,fst_p*2,kernel_size=3,stride=2), # b,16,self.feature_dim*self.max_tree_length//8
            nn.ReLU(True),
            nn.Flatten(1,-1),
        )
        self.out_dim = (max_tree_len//fst_p-1)*fst_p*2
    
    def forward(self, x):
        x = self.tree_layer(x)
        return x

class CNN(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8):
        super(CNN, self).__init__()
        nb = 1
        
        i = max_tree_len/8
        while i/8 > 1:
            i /= 8
            nb += 1

        self.unflatten = nn.Unflatten(1, (1,feature_d*max_tree_len)) # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        for i in range(int(nb)):
            
            if i == nb - 1:
                pool = False
            else:
                pool = True
            cnnbasic = CNN_BasicBlock(inplanes,planes,3,2, pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            if planes <= 8:
                planes *= 2
            else:
                planes = 16
            
        self.tree_layer = nn.Sequential(*bls)
        print(f'##### CNN block {nb} ###### ')
        self.flatten = nn.Flatten(1,-1)
        
        self.out_dim = (feature_d*max_tree_len//(factors)-1)*planes
    
    def forward(self, x):
        x = self.unflatten(x)
        x = self.tree_layer(x)
        x = self.flatten(x)
        return x

class CNN_(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8 ):
        super(CNN_, self).__init__()
        nb = 0
        
        i = max_tree_len/8
        while i/4 > 1:
            i /= 4
            nb += 1
        nb+=1
        self.unflatten = nn.Unflatten(1, (feature_d,max_tree_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        for i in range(int(nb)):
            
            if i == nb - 1:
                pool = False
            else:
                pool = True
            print('CNN_ ',inplanes,planes)
            cnnbasic = CNN_BasicBlock(inplanes,planes,3,2, pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            #if planes <= 8:
            planes *= 2
            #else:
            #    planes = 16
            
        self.tree_layer = nn.Sequential(*bls)
        print(f'##### CNN block {nb} ###### ')
        self.flatten = nn.Flatten(1,-1)
        
        print(f'factors {factors} planes {inplanes}')
        self.out_dim = (max_tree_len//(factors)-1)*inplanes
    
    def forward(self, x):
        #print('input shape', x.shape)
        #x = self.unflatten(x)
        x = self.tree_layer(x)
        x = self.flatten(x)
        return x
        
class CNN_GLO(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8 ):
        super().__init__()
        nb = 0
        
        i = max_tree_len/8
        while i/4 > 1:
            i /= 4
            nb += 1
        nb+=1
        self.unflatten = nn.Unflatten(1, (feature_d,max_tree_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        for i in range(int(nb)):
            if i == nb - 1:
                pool = False
            else:
                pool = True
            print('CNN_ ',inplanes,planes)
            cnnbasic = CNN_BasicBlock(inplanes,planes,3,2, pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            planes *= 2
            
        self.tree_layer = nn.Sequential(*bls)
        
        print(f'##### CNN block {nb} ###### ')
        self.flatten = nn.Flatten(1,-1)
        
        print(f'factors {factors} planes {inplanes}')
        self.out_dim = (max_tree_len//(factors)-1)
    
    def forward(self, x):
        #print('input shape', x.shape)
        #x = self.unflatten(x)
        x = self.tree_layer(x)
        #x = self.flatten(x)
        x = torch.mean(x, 1)
        return x

class CNN_AVG(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8, dropout=True):
        super().__init__()
        nb = 0
        
        i = max_tree_len/8
        while i/4 > 1:
            i /= 4
            nb += 1
        nb+=1
        self.unflatten = nn.Unflatten(1, (feature_d,max_tree_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        for i in range(int(nb)):
            if i == nb - 1:
                pool = False
            else:
                pool = True
            print('CNN_ ',inplanes,planes)
            cnnbasic = CNN_BasicBlock(inplanes,planes,3,2, pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            planes *= 2
        
        bls += [nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1,-1)]

        if dropout:
            bls += [nn.Dropout(0.5)]

        self.tree_layer = nn.Sequential(*bls)
        print(f'##### CNN_AVG block {nb} max_tree_len {max_tree_len} ###### ')
        self.out_dim = inplanes
    
    def forward(self, x):
        x = self.tree_layer(x)
        return x

class CNN_DEP(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8, dropout=True):
        super().__init__()
        nb = 0
        
        i = max_tree_len/4
        while i/4 > 1:
            i /= 4
            nb += 1
        nb+=1
        self.unflatten = nn.Unflatten(1, (feature_d,max_tree_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        for i in range(int(nb)):
            if i == nb - 1:
                pool = False
            else:
                pool = True
            print('CNN_ ',inplanes,planes)
            cnnbasic = CNN_BasicBlock(inplanes,planes,3,2, pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            planes *= 2
        
        bls += [nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1,-1)]

        if dropout:
            bls += [nn.Dropout(0.5)]

        self.tree_layer = nn.Sequential(*bls)
        print(f'##### CNN_AVG block {nb} max_tree_len {max_tree_len} ###### ')
        self.out_dim = inplanes
    
    def forward(self, x):
        x = self.tree_layer(x)
        return x

class CNN_DP(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8, dropout=True):
        super().__init__()
        nb = 0
        
        i = max_tree_len/4
        while i/4 > 1:
            i /= 4
            nb += 1
        nb+=1
        self.unflatten = nn.Unflatten(1, (feature_d,max_tree_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        for i in range(int(nb)):
            if i == nb - 1:
                pool = False
            else:
                pool = True
            #print('CNN_ ',inplanes,planes)
            cnnbasic = CNN_BasicBlock(inplanes,planes,3,2,1,pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            planes *= 2
        
        bls += [nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1,-1)]

        if dropout:
            bls += [nn.Dropout(0.5)]

        self.tree_layer = nn.Sequential(*bls)
        print(f'##### CNN_AVG block {nb} max_tree_len {max_tree_len} ###### ')
        self.out_dim = inplanes
    
    def forward(self, x):
        x = self.tree_layer(x)
        return x

class CNN_TS(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8, dropout=True):
        super().__init__()
        nb = 0
        
        i = max_tree_len/4
        while i/4 > 1:
            i /= 4
            nb += 1
        nb+=1
        self.unflatten = nn.Unflatten(1, (feature_d,max_tree_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        for i in range(int(nb)):
            if i == nb - 1:
                pool = False
            else:
                pool = True
            #print('CNN_ ',inplanes,planes)
            if i == 0:
                kernel, stride, pad = 8,2,3
            elif i == 1:
                kernel, stride, pad = 5,2,2
            else:
                kernel, stride, pad = 3,2,1
            cnnbasic = CNN_BasicBlock(inplanes,planes,kernel,stride,pad,pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            planes *= 2
        
        bls += [nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1,-1)]

        if dropout:
            bls += [nn.Dropout(0.5)]

        self.tree_layer = nn.Sequential(*bls)
        print(f'##### CNN_AVG block {nb} max_tree_len {max_tree_len} ###### ')
        self.out_dim = inplanes
    
    def forward(self, x):
        x = self.tree_layer(x)
        return x

class CNN_OS(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8, dropout=True):
        super().__init__()
        nb = 0
        
        i = max_tree_len/4
        while i/4 > 1:
            i /= 4
            nb += 1
        nb+=1
        self.unflatten = nn.Unflatten(1, (feature_d,max_tree_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1

        for i in range(int(nb)):
            if i == nb - 1:
                pool = False
            else:
                pool = True
            #print('CNN_ ',inplanes,planes)
            if i == 0:
                kernel, stride, pad = 9,1,4
            elif i == 1:
                kernel, stride, pad = 5,1,2
            else:
                kernel, stride, pad = 3,1,1
            cnnbasic = CNN_BasicBlock(inplanes,planes,kernel,stride,pad,pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            planes *= 2
        
        bls += [nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1,-1)]

        if dropout:
            bls += [nn.Dropout(0.5)]

        self.tree_layer = nn.Sequential(*bls)
        print(f'##### CNN_AVG block {nb} max_tree_len {max_tree_len} ###### ')
        self.out_dim = inplanes
    
    def forward(self, x):
        x = self.tree_layer(x)
        return x

class CNN_OK(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8, dropout=True):
        super().__init__()
        nb = 0
        
        i = max_tree_len/4
        while i/4 > 1:
            i /= 4
            nb += 1
        nb+=1
        self.unflatten = nn.Unflatten(1, (feature_d,max_tree_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        k_ = 3+(nb-1)*2
        for i in range(int(nb)):
            if i == nb - 1:
                pool = False
            else:
                pool = True
            #print('CNN_ ',inplanes,planes)
            kernel, stride, pad = k_,1,(k_-1)//2
            cnnbasic = CNN_BasicBlock(inplanes,planes,kernel,stride,pad,pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            planes *= 2
            k_ -= 2
        
        bls += [nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1,-1)]

        if dropout:
            bls += [nn.Dropout(0.5)]

        self.tree_layer = nn.Sequential(*bls)
        print(f'##### CNN_AVG block {nb} max_tree_len {max_tree_len} ###### ')
        self.out_dim = inplanes
    
    def forward(self, x):
        x = self.tree_layer(x)
        return x

class CNN_TK(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8, dropout=True):
        super().__init__()
        nb = 0
        
        i = max_tree_len/4
        while i/4 > 1:
            i /= 4
            nb += 1
        nb+=1
        self.unflatten = nn.Unflatten(1, (feature_d,max_tree_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        k_ = 3+(nb-1)*2
        for i in range(int(nb)):
            if i == nb - 1:
                pool = False
            else:
                pool = True
            #print('CNN_ ',inplanes,planes)
            kernel, stride, pad = k_,2,(k_-1)//2
            cnnbasic = CNN_BasicBlock(inplanes,planes,kernel,stride,pad,pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            planes *= 2
            k_ -= 2
        
        bls += [nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1,-1)]

        if dropout:
            bls += [nn.Dropout(0.5)]

        self.tree_layer = nn.Sequential(*bls)
        print(f'##### CNN_AVG block {nb} max_tree_len {max_tree_len} ###### ')
        self.out_dim = inplanes
    
    def forward(self, x):
        x = self.tree_layer(x)
        return x

class CNN_TKF(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8, dropout=True):
        super().__init__()
        nb = 0
        
        i = max_tree_len/4
        while i/4 > 1:
            i /= 4
            nb += 1
        nb = 4
        self.unflatten = nn.Unflatten(1, (feature_d,max_tree_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        k_ = 3+(nb-1)*2
        for i in range(int(nb)):
            if i == nb - 1:
                pool = False
            else:
                pool = True
            #print('CNN_ ',inplanes,planes)
            kernel, stride, pad = k_,2,(k_-1)//2
            cnnbasic = CNN_BasicBlock(inplanes,planes,kernel,stride,pad,pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            planes *= 2
            k_ -= 2
        
        bls += [nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1,-1)]

        if dropout:
            bls += [nn.Dropout(0.5)]

        self.tree_layer = nn.Sequential(*bls)
        print(f'##### CNN_AVG block {nb} max_tree_len {max_tree_len} ###### ')
        self.out_dim = inplanes
    
    def forward(self, x):
        x = self.tree_layer(x)
        return x



class CNN_FIX(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=8, dropout=True):
        super().__init__()
        nb = 0
        
        nb = 3
        self.unflatten = nn.Unflatten(1, (feature_d,max_tree_len))  # b,1,self.feature_dim*self.max_tree_length
        
        inplanes = feature_d
        planes = fst_p
        bls = []
        factors = 1
        for i in range(int(nb)):
            if i == nb - 1:
                pool = False
            else:
                pool = True
            print('CNN_ ',inplanes,planes)
            cnnbasic = CNN_BasicBlock(inplanes,planes,3,2, pool=pool)
            factors *= cnnbasic.factor
            bls += [cnnbasic]
            inplanes = planes
            planes *= 2
            
        bls += [nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1,-1)]

        if dropout:
            bls += [nn.Dropout(0.5)]
        
        self.tree_layer = nn.Sequential(*bls)
        print(f'##### CNN block {nb} ###### ')

        self.out_dim = inplanes
    
    def forward(self, x):
        x = self.tree_layer(x)
        return x

class RNN(nn.Module):
    def __init__(self, feature_d, max_tree_len):
        super().__init__()
        self.tree_hidden_dim = 16
        self.lstm1_num_layers = 2
        self.tree_layer = nn.Sequential(
            #nn.Unflatten(1,(max_tree_len,feature_d)),
            nn.LSTM(feature_d, self.tree_hidden_dim,num_layers=self.lstm1_num_layers,batch_first=True),
        )
        # self.out_dim = self.tree_hidden_dim
        self.out_dim = max_tree_len

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        lstmout, (hn, cn) = self.tree_layer(x)
        # get the last hidden state (batch, hidden_dim)
        #print('out ',lstmout.shape)
        out = torch.mean(lstmout, 2)
        
        #print('out ',out.shape)
        return out

class GRU(nn.Module):
    def __init__(self, feature_d, max_tree_len):
        super().__init__()
        self.tree_hidden_dim = 100
        self.lstm1_num_layers = 2
        self.tree_layer = nn.Sequential(
            nn.Unflatten(1,(max_tree_len,feature_d)),
            nn.GRU(feature_d, self.tree_hidden_dim,num_layers=self.lstm1_num_layers,batch_first=True),
        )
        # self.out_dim = self.tree_hidden_dim
        self.out_dim = self.tree_hidden_dim

    def forward(self, x):
        out, (hn, cn) = self.tree_layer(x)
        # get the last hidden state (batch, hidden_dim)
        #print('out ',out.shape, 'fn', hn.shape, 'cn', cn.shape)
        out = torch.mean(out, 1)
        
        #print('out ',out.shape)
        return out

class FCN(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, fst_p=32):
        super(FCN, self).__init__()
        
        self.tree_layer = nn.Sequential(
            #nn.Unflatten(1, (1,feature_d*max_tree_len)),
            nn.Conv1d(feature_d,fst_p,kernel_size=9,stride=1,padding=4),
            nn.BatchNorm1d(fst_p),
            nn.ReLU(True),
            nn.Conv1d(fst_p,fst_p*2,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm1d(fst_p*2),
            nn.ReLU(True),
            nn.Conv1d(fst_p*2,fst_p*4,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(fst_p*4),
            nn.ReLU(True),
            #nn.Flatten(1,-1)
        )
        #self.out_dim = int(np.ceil(feature_d*max_tree_len/8))
        self.out_dim = int(np.ceil(max_tree_len))
    

    def block(self, inplanes,planes):
        cnn_block = nn.Sequential(
            nn.Conv1d(inplanes,planes,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(True),
        )
        return cnn_block

    def forward(self, x):
        x = self.tree_layer(x)
        x = torch.mean(x,1)
        return x

class PTCNN_C(pl.LightningModule):
    def __init__(self, feature_d, max_tree_len, nclass=4, fst_c=8, pool='adaptive', method='PTCNN', max_epoch=50):
        super(PTCNN_C, self).__init__()
        if method == 'PTCNN':
            self.feature_map = PTCNN(feature_d, max_tree_len, fst_c, blocks=2, pool=pool)
        elif method == 'CNN':
            self.feature_map = CNN_(feature_d, max_tree_len,fst_p=fst_c)
        elif method == 'CNNGLO':
            self.feature_map = CNN_GLO(feature_d, max_tree_len,fst_p=fst_c)
        elif method == 'CNNAVG':
            self.feature_map = CNN_AVG(feature_d, max_tree_len,fst_p=fst_c)
        elif method == 'CNNFIX':
            self.feature_map = CNN_FIX(feature_d, max_tree_len,fst_p=fst_c)
        elif method == 'CNNDEP':
            self.feature_map = CNN_DEP(feature_d, max_tree_len,fst_p=fst_c)
        elif method == 'CNNTS':
            self.feature_map = CNN_TS(feature_d, max_tree_len,fst_p=fst_c)
        elif method == 'CNNOS':
            self.feature_map = CNN_OS(feature_d, max_tree_len,fst_p=fst_c)
        elif method == 'CNNOK':
            self.feature_map = CNN_OK(feature_d, max_tree_len,fst_p=fst_c)
        elif method == 'CNNTK':
            self.feature_map = CNN_TK(feature_d, max_tree_len,fst_p=fst_c)
        elif method == 'FCN':
            self.feature_map = FCN(feature_d, max_tree_len)
        elif method == 'RNN':
            self.feature_map = RNN(feature_d, max_tree_len)
        elif method == 'GRU':
            self.feature_map = GRU(feature_d, max_tree_len)
        elif method == 'CNNOri':
            self.feature_map = CNNOri(feature_d, max_tree_len, fst_c)
        else:
            raise ValueError(f'Method value error {method}')

        self.max_epoch = max_epoch
        self.fc = nn.Sequential(nn.Dropout(p=0.7),
                                nn.Linear(self.feature_map.out_dim,nclass))
        #self.fc = nn.Linear(self.feature_map.out_dim,nclass)
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.out_dim = nclass
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, tree, label = batch
        y_hat = self.forward(tree)
        loss = F.cross_entropy(y_hat, label)
        a, y_hat = torch.max(y_hat, dim=1)
        
        self.log(f'train_loss', loss, sync_dist=True, on_epoch=True,on_step=False, prog_bar=True)
        self.log('train_acc', self.train_acc(y_hat,label), on_step=False,on_epoch=True,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, tree, label = batch
        y_hat = self.forward(tree)
        loss = F.cross_entropy(y_hat, label)
        a, y_hat = torch.max(y_hat, dim=1)
        self.log(f'val_loss', loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.valid_acc(y_hat,label),on_step=False,on_epoch=True, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, tree, label = batch
        y_hat = self.forward(tree)
        a, y_hat = torch.max(y_hat, dim=1)
        self.test_acc(y_hat,label)
        return {'test_label': label, 'test_pred': y_hat}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        precision, recall, fscore, support = self.epoch_end(outputs,'test')
        return {'test_acc':self.test_acc.compute().item(), 'precision':precision, 'recall':recall, 'fscore':fscore, 'support': support}

    def epoch_end(self, outputs, phase):
        label = torch.cat([x[f'{phase}_label'] for x in outputs]).cpu().detach().numpy()
        pred = torch.cat([x[f'{phase}_pred'] for x in outputs]).cpu().detach().numpy()
        precision, recall, fscore, support = score(label,pred)
        return precision, recall, fscore, support

    def forward(self, x):
        x = self.feature_map(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        #optimizer = Adam(self.parameters(),lr=1e-3,weight_decay=1e-5)
        optimizer = AdamW([
                        {'params': self.parameters(), 'lr':1e-3,'weight_decay':1e-2}
                    ],eps=1e-8)
        #scheduler = StepLR(optimizer, step_size=7,gamma=0.1)
        # scheduler_cosine = get_linear_schedule_with_warmup(optimizer
        #         ,num_warmup_steps=4,num_training_steps=500)

        scheduler_cosineAnneal = CosineAnnealingLR(optimizer,T_max=self.max_epoch)
        scheduler = {
            'scheduler': scheduler_cosineAnneal,
            'name': 'lr_scheduler',
        }
        return [optimizer], [scheduler]


if __name__ == '__main__':
    
    for x in range(100,1001,100):
        if x != 100:
            pass

        '''
        model = PTCNN(1, x, 32, pool='global')
        print(model.out_dim)
        #print(model)
        input = torch.empty((32,x))
        output = model(input)
        print(f'x {x} input {input.shape} output {output.shape}')
        '''
        input = torch.empty((32,6,x))
        m = CNN_TK(6,x, fst_p=32)
        #m = PTCNN_C(6,x,int(32),pool='global',method='FCN')
        output = m(input)
        print(f'x {x} input {input.shape} output {output.shape} out_dim {m.out_dim}')
        