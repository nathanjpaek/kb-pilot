import torch
from torch import nn
import torch.nn.functional as F


class p_model(nn.Module):
    """
    input: N * C * W * H
    output: N * 1 * W * H
    """

    def __init__(self):
        super(p_model, self).__init__()

    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.avg_pool1d(x, c)
        return pooled.view(n, 1, w, h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
