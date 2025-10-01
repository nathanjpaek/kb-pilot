import torch
import torch.nn as nn
import torch.utils.data


class CDM(nn.Module):
    """
    Implementation of the CDM choice model as a Pytorch module
    """

    def __init__(self, n, d):
        """
        Initializes a CDM model

        Args:
        n- number of items in the universe
        d- number of dimensions for feature and context embeddings
        """
        super(CDM, self).__init__()
        self.fs = nn.Parameter(torch.nn.init.normal(torch.Tensor(n, d)))
        self.cs = nn.Parameter(torch.nn.init.normal(torch.Tensor(d, n)))
        self.m = nn.LogSoftmax()
        self.d = d
        self.n = n

    def forward(self, x):
        """
        computes the CDM choice probabilities P(.,S)

        Args:
        x- indicator vector for choice set S, i.e. a 'size(S)-hot' encoding of the
        choice set, or a batch of these
        """
        u = x * torch.sum(torch.mm(self.fs, x * self.cs), dim=1) + (1 - x
            ) * -16
        p = self.m(u)
        return p

    def __str__(self):
        return 'CDM-d=' + str(self.d)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n': 4, 'd': 4}]
