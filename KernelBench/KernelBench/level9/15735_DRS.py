import torch
import torch.nn as nn


class DRS(nn.Module):
    """ 
    DRS non-learnable setting
    hyperparameter O , additional training paramters X
    """

    def __init__(self, delta):
        super(DRS, self).__init__()
        self.relu = nn.ReLU()
        self.delta = delta
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.relu(x)
        """ 1: max extractor """
        x_max = self.global_max_pool(x).view(b, c, 1, 1)
        x_max = x_max.expand_as(x)
        """ 2: suppression controller"""
        control = self.delta
        """ 3: suppressor"""
        x = torch.min(x, x_max * control)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'delta': 4}]
