import torch
import torch.nn as nn


class FcCat(nn.Module):

    def __init__(self, nIn, nOut):
        super(FcCat, self).__init__()
        self.fc = nn.Linear(nIn, nOut, bias=False)

    def forward(self, x):
        out = torch.cat((x, self.fc(x)), 1)
        return out


class Net(nn.Module):

    def __init__(self, nFeatures, nHidden1, nHidden2):
        super(Net, self).__init__()
        self.l1 = FcCat(nFeatures, nHidden1)
        self.l2 = FcCat(nFeatures + nHidden1, nHidden2)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'nFeatures': 4, 'nHidden1': 4, 'nHidden2': 4}]
