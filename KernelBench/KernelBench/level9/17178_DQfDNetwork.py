import torch
import torch.nn as nn
import torch.nn.functional as F


class DQfDNetwork(nn.Module):

    def __init__(self, in_size, out_size):
        super(DQfDNetwork, self).__init__()
        HIDDEN_SIZE = 30
        self.f1 = nn.Linear(in_size, HIDDEN_SIZE)
        self.f2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.f3 = nn.Linear(HIDDEN_SIZE, out_size)
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)
        nn.init.xavier_uniform_(self.f3.weight)
        self.opt = torch.optim.Adam(self.parameters())
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        x1 = F.relu(self.f1(x))
        x2 = F.relu(self.f2(x1))
        x3 = self.f3(x2)
        res = F.softmax(x3)
        return res


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
