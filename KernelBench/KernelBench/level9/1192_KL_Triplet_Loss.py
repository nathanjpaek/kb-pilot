import torch
import torch.nn as nn


class KL_Triplet_Loss(nn.Module):

    def __init__(self, symmetric=True):
        """
        :param symmetric: if symmetric, we will use JS Divergence, if not KL Divergence will be used.
        """
        super().__init__()
        self.symmetric = symmetric
        self.engine = nn.KLDivLoss()

    def forward(self, x, y):
        if len(x.shape) == 4 and len(y.shape) == 4:
            x = x.view(x.size(0) * x.size(1), -1)
            y = y.view(y.size(0) * y.size(1), -1)
        elif len(x.shape) == 2 and len(y.shape) == 2:
            pass
        else:
            raise TypeError('We need a tensor of either rank 2 or rank 4.')
        if self.symmetric:
            loss = self.engine(x, y)
        else:
            loss = self.engine(x, y) + self.engine(y, x)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
