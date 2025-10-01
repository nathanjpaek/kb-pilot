import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class TwoLayer(nn.Module):

    def __init__(self, inputSize, hiddenSize, outputSize):
        super(TwoLayer, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'inputSize': 4, 'hiddenSize': 4, 'outputSize': 4}]
