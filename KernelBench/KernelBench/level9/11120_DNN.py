import torch
import torch.nn as nn
from torch.nn import functional as F


class DNN(nn.Module):

    def __init__(self, n_state, n_action):
        super(DNN, self).__init__()
        self.input_layer = nn.Linear(n_state, 64)
        self.input_layer.weight.data.normal_(0, 0.1)
        self.middle_layer = nn.Linear(64, 32)
        self.middle_layer.weight.data.normal_(0, 0.1)
        self.middle_layer_2 = nn.Linear(32, 32)
        self.middle_layer_2.weight.data.normal_(0, 0.1)
        self.adv_layer = nn.Linear(32, n_action)
        self.adv_layer.weight.data.normal_(0, 0.1)
        self.value_layer = nn.Linear(32, 1)
        self.value_layer.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.middle_layer(x))
        x = F.relu(self.middle_layer_2(x))
        value = self.value_layer(x)
        adv = self.adv_layer(x)
        out = value + adv - adv.mean()
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_state': 4, 'n_action': 4}]
