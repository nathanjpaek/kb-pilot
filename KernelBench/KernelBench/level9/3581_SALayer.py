import torch
import torch.nn as nn
import torch.utils.model_zoo


class SALayer(nn.Module):

    def __init__(self, channel, kernel_size=3):
        super(SALayer, self).__init__()
        self.conv_sa = nn.Conv2d(channel, channel, kernel_size, padding=1,
            groups=channel)

    def forward(self, x):
        y = self.conv_sa(x)
        return x * y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
