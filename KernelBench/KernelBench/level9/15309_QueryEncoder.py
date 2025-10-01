import torch
from torch import nn
import torch.nn.functional as F


class QueryEncoder(nn.Module):

    def __init__(self, input_size):
        super(QueryEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 10)
        self.fc3 = nn.Linear(10, 8)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
