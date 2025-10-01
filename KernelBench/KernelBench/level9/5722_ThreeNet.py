import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeNet(nn.Module):

    def __init__(self, n_features, e1=2048, e2=1024, e3=640, e4=512, e5=216,
        p=0.4):
        super(ThreeNet, self).__init__()
        self.a1 = nn.Linear(n_features, e1)
        self.a2 = nn.Linear(e1, e2)
        self.a3 = nn.Linear(e2, e3)
        self.a4 = nn.Linear(e3, 2)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.selu(self.dropout(self.a1(x)))
        x = F.selu(self.dropout(self.a2(x)))
        x = F.selu(self.dropout(self.a3(x)))
        x = torch.sigmoid(self.a4(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4}]
