import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *


class Discriminator(nn.Module):

    def __init__(self, outputs_size, K=2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(outputs_size, outputs_size // K, bias=True)
        outputs_size = outputs_size // K
        self.fc2 = nn.Linear(outputs_size, outputs_size // K, bias=True)
        outputs_size = outputs_size // K
        self.fc3 = nn.Linear(outputs_size, 2, bias=True)

    def forward(self, x):
        x = x[:, :, None, None]
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = out.view(out.size(0), -1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'outputs_size': 4}]
