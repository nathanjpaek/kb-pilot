import torch
from torch import nn


class BackwardsNet(nn.Module):

    def __init__(self, h, ydim):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.fc1 = torch.nn.Linear(2 * h, h)
        self.fc2 = torch.nn.Linear(h, ydim)

    def forward(self, phiPrev, phi, atn):
        x = torch.cat((phiPrev, phi), 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        loss = self.loss(x, atn)
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'h': 4, 'ydim': 4}]
