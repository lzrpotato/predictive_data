import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class BasicBlock(nn.Module):
    def __init__(
        self, 
        inplanes: int, 
        planes: int, 
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
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
        
        print('identity shape', identity.shape)
        print('out shape', out.shape)
        out += identity
        out = self.relu(out)

        return out

class PTCNN(nn.Module):
    def __init__(self, feature_d, max_tree_len, fst_c):
        super(PTCNN, self).__init__()
        print('prcnn init')
        inplanes = feature_d
        self.input_layer = nn.Sequential(
            nn.Unflatten(1, (feature_d,max_tree_len)),
            nn.BatchNorm1d(inplanes)
        )
        
        nb = max_tree_len // 64
        layers = []
        for bi in range(nb):
            planes = fst_c * 2 ** bi
            layers += self._make_layer(BasicBlock,inplanes,planes,blocks=2,stride=2)
            inplanes = planes
        
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(planes,4)

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
            layers += [BasicBlock(inplanes, planes)]
        
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.input_layer(x)
        print('input_layer shape',out.shape)
        out = self.layers(out)
        print('layers shape',out.shape)
        out = self.avgpool(out)
        print('avgpool shape',out.shape)
        out = self.fc(out)
        
        return out


if __name__ == '__main__':
    print('hi')
    model = PTCNN(1, 100, 8)
    print(model)
    input = torch.empty((1,100))
    output = model(input)
    print('input {} output {}' % (input.shape, output.shape))