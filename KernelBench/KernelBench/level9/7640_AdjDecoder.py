import torch
from torch import nn
import torch.utils.data


class AdjDecoder(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(AdjDecoder, self).__init__()
        self.decode = nn.Linear(featureSize, hiddenSize)
        self.second = nn.Linear(hiddenSize, hiddenSize)
        self.left = nn.Linear(hiddenSize, featureSize)
        self.right = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, parent_in):
        out = self.decode(parent_in)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        l = self.left(out)
        r = self.right(out)
        l = self.tanh(l)
        r = self.tanh(r)
        return l, r


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'featureSize': 4, 'hiddenSize': 4}]
