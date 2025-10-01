import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn


class ComparisonModule(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Conv2d(2 * dim, dim, kernel_size=(1, 1), padding=0
            )
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, in1, in2):
        out = torch.cat([in1, in2], 1)
        out = F.relu(self.projection(out))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
