import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, 32)
        self.linear2 = nn.Linear(32, 1)
        pass

    def forward(self, net):
        value = net
        net = self.linear1(net)
        net = F.relu(net)
        net = self.linear2(net)
        net = torch.flatten(net, 1)
        net = F.softmax(net, dim=1)
        net = torch.unsqueeze(net, dim=1)
        net = torch.matmul(net, value)
        net = torch.flatten(net, 1)
        return net
    pass


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
