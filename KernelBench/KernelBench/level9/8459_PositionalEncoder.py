import math
import torch


class PositionalEncoder(torch.nn.Module):

    def __init__(self, max_freq, feat_size, dimensionality, base=2):
        super().__init__()
        self.max_freq = max_freq
        self.dimensionality = dimensionality
        self.num_bands = math.floor(feat_size / dimensionality / 2)
        self.base = base
        pad = feat_size - self.num_bands * 2 * dimensionality
        self.zero_pad = torch.nn.ZeroPad2d((pad, 0, 0, 0))

    def forward(self, x):
        x = x / 100
        x = x.unsqueeze(-1)
        device = x.device
        dtype = x.dtype
        scales = torch.logspace(0.0, math.log(self.max_freq / 2) / math.log
            (self.base), self.num_bands, base=self.base, device=device,
            dtype=dtype)
        scales = scales[*((None,) * (len(x.shape) - 1)), Ellipsis]
        x = x * scales * math.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(1)
        enc = self.zero_pad(x)
        return enc


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'max_freq': 4, 'feat_size': 4, 'dimensionality': 4}]
