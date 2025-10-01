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


class MLP_FiLM(nn.Module):

    def __init__(self, cdim, fdim):
        super(MLP_FiLM, self).__init__()
        self.l1 = nn.Linear(fdim, fdim)
        self.l2 = nn.Linear(fdim, fdim)
        self.l3 = nn.Linear(fdim, fdim)
        self.f1 = FiLMNetwork(cdim, fdim)
        self.f2 = FiLMNetwork(cdim, fdim)

    def forward(self, c, x):
        x = self.f1(c, self.l1(x)).tanh()
        x = self.f2(c, self.l2(x)).tanh()
        return self.l3(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cdim': 4, 'fdim': 4}]
