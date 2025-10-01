import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class GlobalDiscriminator(nn.Module):

    def __init__(self, y_size, M_channels):
        super().__init__()
        self.c0 = nn.Conv2d(M_channels, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d(16)
        self.l0 = nn.Linear(32 * 16 * 16 + y_size, 256)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = self.avgpool(h)
        h = h.view(M.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'y_size': 4, 'M_channels': 4}]
