import torch
import torch.nn.functional as F


class ResBlock(torch.nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=(3, 3),
            padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv1(y)
        return F.relu(x + y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
