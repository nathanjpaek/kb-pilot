import torch
import torch.nn as nn


class PredictFC(nn.Module):

    def __init__(self, num_params, num_states, in_channels):
        super(PredictFC, self).__init__()
        self.num_params = num_params
        self.fc_param = nn.Conv2d(in_channels, num_params, kernel_size=1,
            stride=1, padding=0, bias=True)
        self.fc_state = nn.Conv2d(in_channels, num_states, kernel_size=1,
            stride=1, padding=0, bias=True)

    def forward(self, input):
        params = self.fc_param(input)
        state = self.fc_state(input)
        return params, state


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_params': 4, 'num_states': 4, 'in_channels': 4}]
