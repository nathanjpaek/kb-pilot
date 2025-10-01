import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet_300_100(nn.Module):
    """Simple NN with hidden layers [300, 100]

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, save_features=None, bench_model=False):
        super(LeNet_300_100, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300, bias=True)
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)
        self.mask = None

    def forward(self, x):
        x0 = x.view(-1, 28 * 28)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return F.log_softmax(x3, dim=1)


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
