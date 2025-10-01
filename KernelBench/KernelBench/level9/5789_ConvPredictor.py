import torch
import torch.nn as nn


class ConvPredictor(nn.Module):

    def __init__(self, input_dim, output_dim, groups):
        super(ConvPredictor, self).__init__()
        self.feature_maps = input_dim
        self.groups = groups
        self.output_dim = output_dim
        self.conv = nn.Conv1d(in_channels=self.feature_maps, out_channels=
            self.groups * self.output_dim, kernel_size=1, groups=self.groups)

    def forward(self, x):
        x = x.unsqueeze(-1)
        outs = torch.stack(torch.split(self.conv(x), self.output_dim, dim=1))
        return outs.sum(0).reshape(-1, self.output_dim)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'groups': 1}]
