import torch
from torch import nn
from typing import Optional


class LinearStack(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int',
        activation_fn: 'Optional[nn.Module]'=None, n: 'int'=1,
        hidden_features: 'Optional[int]'=None, dropout: 'Optional[float]'=None
        ):
        super().__init__()
        if hidden_features is None or n == 1:
            hidden_features = out_features
        modules = []
        for i in range(n):
            if i == 0:
                modules.append(nn.Linear(in_features=in_features,
                    out_features=hidden_features))
            elif 1 < i < n - 1:
                modules.append(nn.Linear(in_features=hidden_features,
                    out_features=hidden_features))
            else:
                modules.append(nn.Linear(in_features=hidden_features,
                    out_features=out_features))
            if activation_fn is not None:
                modules.append(activation_fn())
            if dropout is not None and dropout > 0:
                modules.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*modules)
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' in n:
                torch.nn.init.zeros_(p)
            elif 'weight' in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.net(x)


class Resample(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int',
        activation_fn: 'nn.Module', trainable_add: 'bool'=True, dropout:
        'Optional[float]'=None):
        super().__init__()
        self.in_features = in_features
        self.trainable_add = trainable_add
        self.out_features = out_features
        self.dropout = dropout
        if self.in_features != self.out_features:
            self.resample = LinearStack(in_features=self.in_features,
                out_features=self.out_features, activation_fn=activation_fn,
                dropout=self.dropout)
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.out_features, dtype=
                torch.float))
            self.gate = nn.Sigmoid()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.in_features != self.out_features:
            x = self.resample(x)
        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'activation_fn': 4}]
