import torch
import torch.nn as nn


class Discrete(nn.Module):

    def __init__(self, num_outputs):
        super(Discrete, self).__init__()

    def forward(self, x):
        probs = nn.functional.softmax(x, dim=0)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.entropy()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_outputs': 4}]
