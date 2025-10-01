import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearThreeDeep(nn.Module):

    def __init__(self, inputSize, hiddenSize1, hiddenSize2, hiddenSize3,
        outputSize):
        super().__init__()
        self.inputLinear = nn.Linear(inputSize, hiddenSize1)
        self.hiddenLinear1 = nn.Linear(hiddenSize1, hiddenSize2)
        self.hiddenLinear2 = nn.Linear(hiddenSize2, hiddenSize3)
        self.outputLinear = nn.Linear(hiddenSize3, outputSize)

    def forward(self, inputNodes):
        hiddenNodesL1 = F.relu(self.inputLinear(inputNodes))
        hiddenNodesL2 = F.relu(self.hiddenLinear1(hiddenNodesL1))
        hiddenNodesL3 = F.relu(self.hiddenLinear2(hiddenNodesL2))
        outputNodes = F.softmax(self.outputLinear(hiddenNodesL3))
        return outputNodes

    def getParameters(self):
        return tuple(self.parameters())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inputSize': 4, 'hiddenSize1': 4, 'hiddenSize2': 4,
        'hiddenSize3': 4, 'outputSize': 4}]
