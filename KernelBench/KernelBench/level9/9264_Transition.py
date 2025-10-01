import torch
import torch.nn as nn


class Transition(nn.Module):

    def __init__(self, in_features, out_features, act_layer=nn.GELU):
        super(Transition, self).__init__()
        self.act = act_layer()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
