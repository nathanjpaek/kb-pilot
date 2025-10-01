import torch
from torch import nn
import torch.utils.data


class SymEncoder(nn.Module):

    def __init__(self, featureSize, symmetrySize, hiddenSize):
        super(SymEncoder, self).__init__()
        self.left = nn.Linear(featureSize, hiddenSize)
        self.right = nn.Linear(symmetrySize, hiddenSize)
        self.second = nn.Linear(hiddenSize, hiddenSize)
        self.third = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, left_in, right_in):
        out = self.left(left_in)
        out += self.right(right_in)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        out = self.third(out)
        out = self.tanh(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'featureSize': 4, 'symmetrySize': 4, 'hiddenSize': 4}]
