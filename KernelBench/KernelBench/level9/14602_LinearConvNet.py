import torch
import torch.nn as nn


class LinearConvNet(nn.Module):

    def __init__(self):
        super(LinearConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.conv2 = nn.Conv2d(1, 3, 2, 1, bias=False)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        output = torch.stack([conv1_out.sum(dim=(1, 2, 3)), conv2_out.sum(
            dim=(1, 2, 3))], dim=1)
        return output


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
