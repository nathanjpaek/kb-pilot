import torch
import torch.nn as nn
from math import sqrt as sqrt
from itertools import product as product


class RON(nn.Module):

    def __init__(self, lat_inC, top_inC, outC):
        super(RON, self).__init__()
        self.latlayer = nn.Conv2d(lat_inC, outC, 3, 1, padding=1)
        self.toplayer = nn.ConvTranspose2d(top_inC, outC, 2, 2)

    def forward(self, bottom, top):
        x = self.latlayer(bottom)
        y = self.toplayer(top)
        return x + y


def get_inputs():
    return [torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lat_inC': 4, 'top_inC': 4, 'outC': 4}]
