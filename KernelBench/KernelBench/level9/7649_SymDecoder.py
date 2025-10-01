import torch
from torch import nn
import torch.utils.data


class SymDecoder(nn.Module):

    def __init__(self, featureSize, symmetrySize, hiddenSize):
        super(SymDecoder, self).__init__()
        self.decode = nn.Linear(featureSize, hiddenSize)
        self.second = nn.Linear(hiddenSize, hiddenSize)
        self.tanh = nn.Tanh()
        self.left = nn.Linear(hiddenSize, featureSize)
        self.right = nn.Linear(hiddenSize, symmetrySize)

    def forward(self, parent_in):
        out = self.decode(parent_in)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        f = self.left(out)
        f = self.tanh(f)
        s = self.right(out)
        s = self.tanh(s)
        return f, s


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'featureSize': 4, 'symmetrySize': 4, 'hiddenSize': 4}]
