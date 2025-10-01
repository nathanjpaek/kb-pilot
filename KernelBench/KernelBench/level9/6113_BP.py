import torch
import torch.nn as nn
import torch.utils.data


class BP(nn.Module):
    """
    Implementation of the Bastell-Polking k-th order model as a pytorch module
    """

    def __init__(self, n, k, d):
        """
        Initializes a k-th order Batsell-Polking model

        Args:
        n- number of items in universe
        k- order of the model
        d- rank of the model
        """
        super(BP, self).__init__()
        shape = tuple([n] * k)
        self.U = nn.Parameter(torch.nn.init.normal_(torch.Tensor(*shape)))
        self.k = k
        self.n = n
        self.d = d
        self.m = nn.LogSoftmax()

    def forward(self, x):
        """
        Computes choice probabilities for k-th order BP model
        """
        utils = self.U
        for _ in range(self.k - 1):
            utils = torch.matmul(utils, torch.squeeze(x))
        utils = x * utils + (1 - x) * -16
        return self.m(utils)

    def __str__(self):
        return 'BP:k=' + str(self.k)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n': 4, 'k': 4, 'd': 4}]
