import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-06):
        """
        Args:
            p : int
                Number of the pooling parameter
            eps : float
                lower-bound of the range to be clamped to
        """
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = torch.clamp(x, min=self.eps)
        x = torch.pow(x, self.p.unsqueeze(-1).unsqueeze(-1))
        x = F.avg_pool2d(x, x.size(-2), x.size(-1))
        x = torch.pow(x, 1.0 / self.p.unsqueeze(-1).unsqueeze(-1))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
