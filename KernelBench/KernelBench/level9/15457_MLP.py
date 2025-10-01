import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        n_hid = 20
        n_out = 10
        self.l1 = nn.Linear(28 * 28, n_hid)
        self.l2 = nn.Linear(n_hid, n_hid)
        self.l3 = nn.Linear(n_hid, n_hid)
        self.l4 = nn.Linear(n_hid, n_out)

    def forward(self, x: 'torch.Tensor'):
        x = x.view([-1, 28 * 28])
        x = F.sigmoid(self.l1(x))
        x = F.sigmoid(self.l2(x))
        x = F.sigmoid(self.l3(x))
        x = self.l4(x)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
