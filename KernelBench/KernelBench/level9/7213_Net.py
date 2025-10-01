import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data


class Net(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 120, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0, std=1)
        self.fc2 = nn.Linear(120, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]
