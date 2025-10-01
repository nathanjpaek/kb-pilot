import torch
import torch.nn as nn
import torch.nn.functional as F


class FRM(nn.Module):

    def __init__(self, nb_dim, do_add=True, do_mul=True):
        super(FRM, self).__init__()
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()
        self.do_add = do_add
        self.do_mul = do_mul

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)
        if self.do_mul:
            x = x * y
        if self.do_add:
            x = x + y
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'nb_dim': 4}]
