import torch
import torch.nn as nn
from matplotlib.font_manager import *


class Categorical(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
