import torch
import torch.nn as nn


class NetEnd(nn.Module):

    def __init__(self, num_classes: 'int'):
        super(NetEnd, self).__init__()
        self.num_classes = num_classes
        self.fc_net1 = nn.Conv2d(21, self.num_classes, kernel_size=1, stride=1)
        assert self.num_classes > 0, 'The number of classes must be a positive integer.'
        if self.num_classes > 1:
            self.final = nn.Softmax()
        else:
            self.final = nn.Sigmoid()

    def forward(self, x):
        out = self.fc_net1(x)
        out = self.final(out)
        return out


def get_inputs():
    return [torch.rand([4, 21, 64, 64])]


def get_init_inputs():
    return [[], {'num_classes': 4}]
