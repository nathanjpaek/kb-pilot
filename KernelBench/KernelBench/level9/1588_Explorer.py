import torch
import numpy as np
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Explorer(nn.Module):

    def __init__(self, state_dim, max_action, exp_regularization):
        super(Explorer, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_
                (x, 0), np.sqrt(2))
        self.l1 = init_(nn.Linear(state_dim, 64))
        self.l2 = init_(nn.Linear(64, 64))
        self.l3 = init_(nn.Linear(64, state_dim))
        self.max_action = max_action
        self.exp_regularization = exp_regularization

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a)
            ) * self.exp_regularization ** 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'max_action': 4, 'exp_regularization': 4}]
