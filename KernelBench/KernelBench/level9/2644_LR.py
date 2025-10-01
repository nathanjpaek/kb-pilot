import torch
from torch import nn
import torch.utils.data


class LR(nn.Module):

    def __init__(self, dimension, num_class=2):
        super(LR, self).__init__()
        self.last_layer = nn.Linear(dimension, num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.last_layer(x)
        x = self.softmax(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dimension': 4}]
