import torch
import torch.nn as nn


def taylor_softmax_v1(x, dim=1, n=4, use_log=False):
    assert n % 2 == 0 and n > 0
    fn = torch.ones_like(x)
    denor = 1.0
    for i in range(1, n + 1):
        denor *= i
        fn = fn + x.pow(i) / denor
    out = fn / fn.sum(dim=dim, keepdims=True)
    if use_log:
        out = out.log()
    return out


class LogTaylorSoftmaxV1(nn.Module):

    def __init__(self, dim=1, n=2):
        super(LogTaylorSoftmaxV1, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        return taylor_softmax_v1(x, self.dim, self.n, use_log=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
