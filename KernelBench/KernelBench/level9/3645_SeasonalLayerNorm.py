import torch
import torch.nn as nn
import torch.fft


class SeasonalLayerNorm(nn.Module):
    """Special designed layernorm for the seasonal part."""

    def __init__(self, channels):
        super(SeasonalLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
