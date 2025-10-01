import torch
import numpy as np
import torch.nn as nn


class Coxnnet(nn.Module):

    def __init__(self, nfeat):
        super(Coxnnet, self).__init__()
        self.fc1 = nn.Linear(nfeat, int(np.ceil(nfeat ** 0.5)))
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(int(np.ceil(nfeat ** 0.5)), 1)
        self.init_hidden()

    def forward(self, x, coo=None):
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def init_hidden(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4}]
