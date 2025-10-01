import torch
from torch import nn
import torch.distributions


class Res(nn.Module):

    def __init__(self, H):
        super().__init__()
        self.u1 = nn.Linear(H, H)
        self.u2 = nn.Linear(H, H)
        self.v1 = nn.Linear(H, H)
        self.v2 = nn.Linear(H, H)
        self.w = nn.Linear(H, H)

    def forward(self, y):
        y = self.w(y)
        y = y + torch.relu(self.v1(torch.relu(self.u1(y))))
        return y + torch.relu(self.v2(torch.relu(self.u2(y))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'H': 4}]
