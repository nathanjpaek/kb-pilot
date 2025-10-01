import torch
import torch.nn as nn
import torch.utils.data
import torch
import torch.nn.parallel


class VNLinear(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
