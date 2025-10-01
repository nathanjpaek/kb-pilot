import torch
from torchvision.transforms import functional as F
import torch.utils.data
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


class BiAttention(nn.Module):

    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0
        ):
        super(BiAttention, self).__init__()
        self.hidden_aug = 3
        self.glimpses = glimpses
        self.lin_v = FCNet(v_features, int(mid_features * self.hidden_aug),
            activate='relu', drop=drop / 2.5)
        self.lin_q = FCNet(q_features, int(mid_features * self.hidden_aug),
            activate='relu', drop=drop / 2.5)
        self.h_weight = nn.Parameter(torch.Tensor(1, glimpses, 1, int(
            mid_features * self.hidden_aug)).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, glimpses, 1, 1).normal_())
        self.drop = nn.Dropout(drop)

    def forward(self, v, q):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        """
        v_num = v.size(1)
        q_num = q.size(1)
        v_ = self.lin_v(v).unsqueeze(1)
        q_ = self.lin_q(q).unsqueeze(1)
        v_ = self.drop(v_)
        h_ = v_ * self.h_weight
        logits = torch.matmul(h_, q_.transpose(2, 3))
        logits = logits + self.h_bias
        atten = F.softmax(logits.view(-1, self.glimpses, v_num * q_num), 2)
        return atten.view(-1, self.glimpses, v_num, q_num)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'v_features': 4, 'q_features': 4, 'mid_features': 4,
        'glimpses': 4}]
