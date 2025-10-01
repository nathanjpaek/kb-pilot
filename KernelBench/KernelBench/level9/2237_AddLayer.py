import torch
from torch import nn
import torch.utils.checkpoint


class AddLayer(nn.Module):

    def __init__(self, t1, t2):
        super(AddLayer, self).__init__()
        self.t1 = t1
        self.t2 = t2

    def forward(self, x, y):
        return x + y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'t1': 4, 't2': 4}]
