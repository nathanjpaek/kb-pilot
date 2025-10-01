import torch
import torch.nn as nn
import torch.nn.parallel


class NoiseInjection(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(0.01 * torch.randn(1, channel, 1, 1))

    def forward(self, feat, noise=None):
        if noise is None:
            noise = torch.randn(feat.shape[0], 1, feat.shape[2], feat.shape
                [3], dtype=feat.dtype, device=feat.device)
        return feat + self.weight * noise


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
