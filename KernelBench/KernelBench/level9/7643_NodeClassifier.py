import torch
from torch import nn
import torch.utils.data


class NodeClassifier(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(NodeClassifier, self).__init__()
        self.first = nn.Linear(featureSize, hiddenSize)
        self.tanh = nn.Tanh()
        self.second = nn.Linear(hiddenSize, 3)
        self.softmax = nn.Softmax()

    def forward(self, feature):
        out = self.first(feature)
        out = self.tanh(out)
        out = self.second(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'featureSize': 4, 'hiddenSize': 4}]
