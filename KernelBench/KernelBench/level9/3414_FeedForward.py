import torch
import torch.nn.functional as F
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, num_classes)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = x.view(x.size(0), 784)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
