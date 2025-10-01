import torch
import torch.nn as nn
import torch.nn.functional as F


class UniverseHead(torch.nn.Module):
    """ universe agent example
        input: [None, 42, 42, 1]; output: [None, 288];
    """

    def __init__(self, n):
        super(UniverseHead, self).__init__()
        self.conv1 = nn.Conv2d(n, 32, kernel_size=(3, 3), stride=(2, 2),
            padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2),
            padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2),
            padding=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2),
            padding=(1, 1))
        self.output_size = 288

    def forward(self, state):
        output = F.elu(self.conv1(state))
        output = F.elu(self.conv2(output))
        output = F.elu(self.conv3(output))
        output = F.elu(self.conv4(output))
        return output.view(-1, self.output_size)


def get_inputs():
    return [torch.rand([4, 4, 48, 48])]


def get_init_inputs():
    return [[], {'n': 4}]
