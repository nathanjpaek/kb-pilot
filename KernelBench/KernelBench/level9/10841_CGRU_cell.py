import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f
from math import sqrt as sqrt
from itertools import product as product


class CGRU_cell(nn.Module):
    """Initialize a basic Conv GRU cell.
    Args:
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
    """

    def __init__(self, input_chans, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = int((filter_size - 1) / 2)
        self.ConvGates = nn.Conv2d(self.input_chans + self.num_features, 2 *
            self.num_features, 3, padding=self.padding)
        self.Conv_ct = nn.Conv2d(self.input_chans + self.num_features, self
            .num_features, 3, padding=self.padding)

    def forward(self, input, hidden_state):
        hidden = hidden_state
        c1 = self.ConvGates(torch.cat((input, hidden), 1))
        rt, ut = c1.chunk(2, 1)
        reset_gate = f.sigmoid(rt)
        update_gate = f.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = f.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h

    def init_hidden(self, input):
        feature_size = input.size()[-2:]
        return Variable(torch.zeros(input.size(0), self.num_features,
            feature_size[0], feature_size[1]))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_chans': 4, 'filter_size': 4, 'num_features': 4}]
