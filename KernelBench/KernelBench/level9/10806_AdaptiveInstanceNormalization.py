import torch
import torch.nn as nn


class AdaptiveInstanceNormalization(nn.Module):
    """Some Information about AdaptiveInstanceNormalization"""

    def __init__(self, channels, style_dim):
        super(AdaptiveInstanceNormalization, self).__init__()
        self.affine = nn.Linear(style_dim, channels * 2)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x, style):
        scale, bias = self.affine(style).chunk(2, dim=1)
        scale = scale.unsqueeze(2).unsqueeze(3)
        bias = bias.unsqueeze(2).unsqueeze(3)
        x = self.norm(x)
        x = x * scale + bias
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'style_dim': 4}]
