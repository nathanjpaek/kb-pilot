import torch
import torch.nn as nn


class HighwayLayer(nn.Module):

    def __init__(self, in_units, out_units):
        super(HighwayLayer, self).__init__()
        self.highway_linear = nn.Linear(in_features=in_units, out_features=
            out_units, bias=True)
        self.highway_gate = nn.Linear(in_features=in_units, out_features=
            out_units, bias=True)

    def forward(self, input_x):
        highway_g = torch.relu(self.highway_linear(input_x))
        highway_t = torch.sigmoid(self.highway_gate(input_x))
        highway_out = torch.mul(highway_g, highway_t) + torch.mul(1 -
            highway_t, input_x)
        return highway_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_units': 4, 'out_units': 4}]
