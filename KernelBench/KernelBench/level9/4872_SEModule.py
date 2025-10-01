import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0)

    def forward(self, x):
        out = x.mean(dim=(2, 3), keepdim=True)
        out = F.relu(self.fc1(out), inplace=True)
        out = torch.sigmoid(self.fc2(out))
        return x * out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'reduction': 4}]
