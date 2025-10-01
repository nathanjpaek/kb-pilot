import torch
import torch.nn as nn


class LinearLR(nn.Module):
    """[u * v + res] version of torch.nn.Linear"""

    def __init__(self, in_features, out_features, rank_ratio=0.25, bias=
        True, device=None, dtype=None):
        super().__init__()
        sliced_rank = int(min(in_features, out_features) * rank_ratio)
        self.u = nn.Linear(in_features, sliced_rank, bias=False, device=
            device, dtype=dtype)
        self.v = nn.Linear(sliced_rank, out_features, bias=bias, device=
            device, dtype=dtype)
        self.res = nn.Linear(in_features, out_features, bias=False, device=
            device, dtype=dtype)

    def freeze(self):
        for param in self.res.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.res.parameters():
            param.requires_grad = True

    def forward(self, input):
        return self.v(self.u(input)) + self.res(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
