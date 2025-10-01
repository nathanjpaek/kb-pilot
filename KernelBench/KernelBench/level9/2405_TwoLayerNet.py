import torch
from torch import nn


class TwoLayerNet(nn.Module):

    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = None
        z1 = self.linear1(torch.flatten(x.clone().detach(), start_dim=1))
        h1 = torch.sigmoid(z1)
        out = self.linear2(h1)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_size': 4, 'num_classes': 4}]
