import torch
import torch.nn as nn


class FiLMNetwork(nn.Module):

    def __init__(self, in_sz, out_sz):
        super(FiLMNetwork, self).__init__()
        self.f = nn.Linear(in_sz, out_sz)
        self.h = nn.Linear(in_sz, out_sz)

    def forward(self, inputs, features):
        gamma = self.f(inputs).unsqueeze(1)
        beta = self.h(inputs).unsqueeze(1)
        return features * gamma + beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_sz': 4, 'out_sz': 4}]
