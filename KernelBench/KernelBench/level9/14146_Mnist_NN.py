import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist_NN(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 512, bias=True)
        self.lin2 = nn.Linear(512, 256, bias=True)
        self.lin3 = nn.Linear(256, 10, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 784)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
