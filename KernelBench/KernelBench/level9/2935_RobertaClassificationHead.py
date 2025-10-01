import torch
import torch.nn as nn
from typing import Optional


class RobertaClassificationHead(nn.Module):

    def __init__(self, num_classes, input_dim, inner_dim: 'Optional[int]'=
        None, dropout: 'float'=0.1, activation=nn.ReLU):
        super().__init__()
        if not inner_dim:
            inner_dim = input_dim
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.activation_fn = activation()

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_classes': 4, 'input_dim': 4}]
