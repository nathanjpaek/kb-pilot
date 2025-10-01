import torch
from torch import nn
from torch.nn import functional as F
import torch.onnx
from torch.optim.lr_scheduler import *


class MnistMLP(nn.Module):

    def __init__(self, hidden_size=500):
        super(MnistMLP, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
