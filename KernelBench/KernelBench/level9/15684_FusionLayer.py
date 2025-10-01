import torch
from torch import nn
from torch.nn import functional as F


class FusionLayer(nn.Module):
    """
        make a fusion two vectors
    """

    def __init__(self, hdim):
        super(FusionLayer, self).__init__()
        self.linear_fusion = nn.Linear(hdim * 4, hdim)
        self.linear_gate = nn.Linear(hdim * 4, 1)

    def forward(self, x1, x2):
        """
        Args:
            x1: bxnxd
            x2: bxnxd

        Returns:
            ret: bxnxd
        """
        m = F.tanh(self.linear_fusion(torch.cat([x1, x2, x1 * x2, x1 - x2],
            dim=2)))
        g = F.sigmoid(self.linear_gate(torch.cat([x1, x2, x1 * x2, x1 - x2],
            dim=2)))
        ret = g * m + (1 - g) * x2
        return ret


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hdim': 4}]
