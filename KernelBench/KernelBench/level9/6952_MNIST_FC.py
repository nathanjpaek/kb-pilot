import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_FC(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, xb):
        xb = xb.view(-1, 28 * 28)
        xb = F.relu(self.fc1(xb))
        xb = F.softmax(self.fc2(xb))
        return xb.view(-1, 10)


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
