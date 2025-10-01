import torch
import torch.nn as nn


class MLP1x(nn.Module):

    def __init__(self, dim, hidd, num_classes=10):
        super(MLP1x, self).__init__()
        self.fc1 = nn.Linear(dim, hidd)
        self.fc2 = nn.Linear(hidd, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'hidd': 4}]
