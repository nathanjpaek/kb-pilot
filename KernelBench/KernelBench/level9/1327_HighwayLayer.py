import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed


def my_xavier_init(m, gain=1):
    """Xavier initialization: weights initialization that tries to make variance of outputs
    of a layer equal to variance of its inputs.
    """
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain)
        else:
            nn.init.constant_(p, 0)


class HighwayLayer(torch.nn.Module):
    """Highway transformation used in span prediction."""

    def __init__(self, dim):
        super(HighwayLayer, self).__init__()
        self.gate_proj = nn.Linear(dim, dim, bias=True)
        self.nlin_proj = nn.Linear(dim, dim, bias=True)
        my_xavier_init(self.nlin_proj)
        my_xavier_init(self.gate_proj)
        nn.init.constant_(self.gate_proj.bias, -1)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        nlin = torch.tanh(self.nlin_proj(x))
        res = gate * nlin + (1 - gate) * x
        return res


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
