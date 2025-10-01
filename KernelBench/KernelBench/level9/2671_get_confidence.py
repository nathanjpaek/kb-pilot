import torch
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse


class get_confidence(nn.Module):

    def __init__(self, num_in_layers, num_out_layers=1):
        super(get_confidence, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=3,
            stride=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = p, p, p, p
        x_out = self.conv1(F.pad(x, p2d))
        return self.sigmoid(x_out)


def get_inputs():
    return [torch.rand([4, 1, 4, 4])]


def get_init_inputs():
    return [[], {'num_in_layers': 1}]
