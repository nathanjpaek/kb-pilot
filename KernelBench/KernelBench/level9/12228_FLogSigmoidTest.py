import torch
import torch.nn as nn


class FLogSigmoidTest(nn.Module):
    """
    Test for nn.functional types
    """

    def __init__(self):
        super(FLogSigmoidTest, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.logsigmoid(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
