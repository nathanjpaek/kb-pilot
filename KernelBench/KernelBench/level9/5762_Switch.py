import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal


def ZeroInitializer(param):
    shape = param.size()
    init = np.zeros(shape).astype(np.float32)
    param.data.set_(torch.from_numpy(init))


def Linear(initializer=kaiming_normal, bias_initializer=ZeroInitializer):


    class CustomLinear(nn.Linear):

        def reset_parameters(self):
            initializer(self.weight)
            if self.bias is not None:
                bias_initializer(self.bias)
    return CustomLinear


class Switch(nn.Module):

    def __init__(self, hidden_dim):
        super(Switch, self).__init__()
        self.fc1 = Linear()(in_features=3 * hidden_dim, out_features=
            hidden_dim, bias=False)
        self.fc2 = Linear()(in_features=hidden_dim, out_features=1, bias=False)

    def forward(self, hl, hr, hn):
        h_cat = torch.cat([hl, hr, hn], dim=2)
        h_tmp = F.tanh(self.fc1(h_cat))
        alpha = F.sigmoid(self.fc2(h_tmp))
        return alpha


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'hidden_dim': 4}]
