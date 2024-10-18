import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from collections import OrderedDict
import numpy as np
import pandas

k = 320
k2 = 384

class Model(nn.Module):
    def __init__(self, columns, targets):
        super(Model, self).__init__()
        self.net1 = Sequential(OrderedDict([
            ('batch norm1', nn.BatchNorm1d(len(columns) - 1, momentum=0.99)),
            ('linear1', nn.Linear(len(columns) - 1, k)),
            ('relu1', nn.ReLU())
        ]))
        self.net2 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear2', nn.Linear(k, k)),
            ('relu2', nn.ReLU())
        ]))
        self.net10 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear2', nn.Linear(k, k2)),
            ('relu2', nn.ReLU())
        ]))
        self.net11 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k2, momentum=0.99)),
            ('linear2', nn.Linear(k2, k2)),
            ('relu2', nn.ReLU())
        ]))
        self.net12 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k2, momentum=0.99)),
            ('linear2', nn.Linear(k2, k)),
            ('relu2', nn.ReLU())
        ]))
        self.net13 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear2', nn.Linear(k, k)),
            ('relu2', nn.ReLU())
        ]))
        self.net3 = Sequential(OrderedDict([
            ('batch norm3', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear3', nn.Linear(k, targets)),
            ('softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, x):
        a = self.net1(x)
        b = self.net2(a) + a
        h1 = self.net10(b)
        h2 = self.net11(h1)
        h3 = self.net12(h2)
        h4 = self.net13(h3)
        c = self.net3(h4)
        c = c.squeeze(-1)
        return c