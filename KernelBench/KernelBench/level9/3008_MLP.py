from torch.nn import Module
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from scipy.sparse import *


class MLP(Module):

    def __init__(self, features_dim, hidden_dim, out_dim, bias=True,
        dropout=0.3):
        super(MLP, self).__init__()
        self.features_dim = features_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.linear = torch.nn.Linear(features_dim, hidden_dim)
        self.z_mean = torch.nn.Linear(hidden_dim, out_dim)
        self.z_log_std = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, input):
        hidden = F.relu(self.linear(input))
        z_mean = F.dropout(self.z_mean(hidden), self.dropout, training=self
            .training)
        z_log_std = F.dropout(self.z_log_std(hidden), self.dropout,
            training=self.training)
        return z_mean, z_log_std

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features_dim': 4, 'hidden_dim': 4, 'out_dim': 4}]
