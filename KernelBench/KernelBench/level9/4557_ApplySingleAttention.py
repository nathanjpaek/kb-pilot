import torch
import torch.utils.data
import torch.nn as nn
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


class ApplySingleAttention(nn.Module):

    def __init__(self, v_features, q_features, mid_features, drop=0.0):
        super(ApplySingleAttention, self).__init__()
        self.lin_v = FCNet(v_features, mid_features, activate='relu', drop=drop
            )
        self.lin_q = FCNet(q_features, mid_features, activate='relu', drop=drop
            )
        self.lin_atten = FCNet(mid_features, mid_features, drop=drop)

    def forward(self, v, q, atten):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        atten:  batch x v_num x q_num
        """
        v_ = self.lin_v(v).transpose(1, 2).unsqueeze(2)
        q_ = self.lin_q(q).transpose(1, 2).unsqueeze(3)
        v_ = torch.matmul(v_, atten.unsqueeze(1))
        h_ = torch.matmul(v_, q_)
        h_ = h_.squeeze(3).squeeze(2)
        atten_h = self.lin_atten(h_.unsqueeze(1))
        return atten_h


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'v_features': 4, 'q_features': 4, 'mid_features': 4}]
