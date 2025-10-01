import torch
import torch.nn as nn
import torch.utils.data


class MNL(nn.Module):
    """
    Implementation of MNL choice model as a Pytorch module
    """

    def __init__(self, n):
        super(MNL, self).__init__()
        self.u = nn.Parameter(torch.nn.init.normal(torch.Tensor(n)))
        self.n = n
        self.m = nn.Softmax()

    def forward(self, x):
        """
        computes the PCMC choice probabilities P(.,S)

        Args:
        x- indicator vector for choice set S, i.e. a 'size(S)-hot' encoding of the
        choice set, or a batch of such encodings
        """
        u = x * self.u + (1 - x) * -16
        p = self.m(u)
        return torch.log(p / torch.sum(p))

    def __str__(self):
        return 'MNL'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n': 4}]
