import torch
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class FCNet(nn.Module):

    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)
        self.drop_value = drop
        self.drop = nn.Dropout(drop)
        self.activate = activate.lower() if activate is not None else None
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        x = self.lin(x)
        if self.activate is not None:
            x = self.ac_fn(x)
        return x


class Attention(nn.Module):

    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0
        ):
        super(Attention, self).__init__()
        self.lin_v = FCNet(v_features, mid_features, activate='relu')
        self.lin_q = FCNet(q_features, mid_features, activate='relu')
        self.lin = FCNet(mid_features, glimpses, drop=drop)

    def forward(self, v, q):
        """
        v = batch, num_obj, dim
        q = batch, dim
        """
        v = self.lin_v(v)
        q = self.lin_q(q)
        batch, num_obj, _ = v.shape
        _, q_dim = q.shape
        q = q.unsqueeze(1).expand(batch, num_obj, q_dim)
        x = v * q
        x = self.lin(x)
        x = F.softmax(x, dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'v_features': 4, 'q_features': 4, 'mid_features': 4,
        'glimpses': 4}]
