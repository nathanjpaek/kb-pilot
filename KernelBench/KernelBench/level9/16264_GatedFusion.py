import torch
import torch.nn as nn
from scipy.sparse import *


class GatedFusion(nn.Module):

    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        """GatedFusion module"""
        self.fc_z = nn.Linear(4 * hidden_size, hidden_size, bias=True)

    def forward(self, h_state, input):
        z = torch.sigmoid(self.fc_z(torch.cat([h_state, input, h_state *
            input, h_state - input], -1)))
        h_state = (1 - z) * h_state + z * input
        return h_state


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
