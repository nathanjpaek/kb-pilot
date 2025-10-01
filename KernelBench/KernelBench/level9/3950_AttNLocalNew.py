import torch
import torch.nn as nn


class AttNLocalNew(nn.Module):
    """
    自动限制矩阵

    实现斜对角线保留权重，其他的设为-inf


    """

    def __init__(self, maxlen=128, limit=20):
        super(AttNLocalNew, self).__init__()
        self.limit = limit
        self.maxlen = maxlen
        pass

    def forward(self, x):
        mask = torch.ones_like(x).tril(diagonal=-1) + torch.ones_like(x).triu(
            diagonal=self.limit)
        x[mask == 1] = -float('Inf')
        return x
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
