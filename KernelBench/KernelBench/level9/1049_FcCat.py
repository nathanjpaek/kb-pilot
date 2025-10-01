import torch
import torch.nn as nn


class FcCat(nn.Module):

    def __init__(self, nIn, nOut):
        super(FcCat, self).__init__()
        self.fc = nn.Linear(nIn, nOut, bias=False)

    def forward(self, x):
        out = torch.cat((x, self.fc(x)), 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nIn': 4, 'nOut': 4}]
