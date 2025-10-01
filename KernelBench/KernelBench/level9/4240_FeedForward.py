import math
import torch
from torch import nn


class GELU(nn.Module):

    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, tensor):
        geluPow = tensor + 0.044715 * torch.pow(tensor, 3)
        geluTanh = torch.tanh(math.sqrt(2 / math.pi) * geluPow)
        geluResult = 1 + geluTanh
        return 0.5 * tensor * geluResult


class FeedForward(nn.Module):

    def __init__(self, hiddenSize, innerLayerDimension, dropOutProb=0.1):
        super(FeedForward, self).__init__()
        self.activationFuncion = GELU()
        self.dropout = nn.Dropout(dropOutProb)
        self.w1 = nn.Linear(hiddenSize, innerLayerDimension)
        self.w2 = nn.Linear(innerLayerDimension, hiddenSize)

    def forward(self, tensor):
        intermediate = self.activationFuncion(self.w1(tensor))
        linearOut = self.w2(intermediate)
        return self.dropout(linearOut)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hiddenSize': 4, 'innerLayerDimension': 4}]
