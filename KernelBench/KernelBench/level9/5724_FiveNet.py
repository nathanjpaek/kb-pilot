import torch
import torch.nn as nn
import torch.nn.functional as F


class FiveNet(nn.Module):

    def __init__(self, n_features, e1=1024, e2=2048, e3=1024, e4=640, e5=
        512, p=0.4):
        super(FiveNet, self).__init__()
        self.a1 = nn.Linear(n_features, e2)
        self.a2 = nn.Linear(e2, e3)
        self.a3 = nn.Linear(e3, e4)
        self.a4 = nn.Linear(e4, e5)
        self.a5 = nn.Linear(e5, 2)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.selu(self.dropout(self.a1(x)))
        x = F.selu(self.dropout(self.a2(x)))
        x = F.selu(self.dropout(self.a3(x)))
        x = F.selu(self.dropout(self.a4(x)))
        x = torch.sigmoid(self.a5(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4}]
