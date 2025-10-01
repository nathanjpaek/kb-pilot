import torch
import numpy as np
import torch.optim


def _acos_safe(x: 'torch.Tensor', eps: 'float'=0.0001):
    slope = np.arccos(1.0 - eps) / eps
    buf = torch.empty_like(x)
    good = torch.abs(x) <= 1.0 - eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = torch.acos(sign * (1.0 - eps)) - slope * sign * (torch.abs(x
        [bad]) - 1.0 + eps)
    return buf


class CosineDistance(torch.nn.CosineSimilarity):

    def __init__(self, dim: 'int'=1, epsilon: 'float'=0.0001, normalized:
        'bool'=True):
        super(CosineDistance, self).__init__(dim=dim, eps=epsilon)
        self.normalized = normalized
        self.epsilon = epsilon

    def forward(self, gt: 'torch.Tensor', pred: 'torch.Tensor', weights:
        'torch.Tensor'=None, mask: 'torch.Tensor'=None) ->torch.Tensor:
        dot = torch.sum(gt * pred, dim=self.dim) if self.normalized else super(
            CosineDistance, self).forward(gt, pred)
        return _acos_safe(dot, eps=self.epsilon) / np.pi


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
