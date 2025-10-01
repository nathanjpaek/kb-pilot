import torch
import numpy as np
from torch import nn


class Nloss_GD(nn.Module):

    def __init__(self, dim):
        super(Nloss_GD, self).__init__()
        self.dim = dim
        torch.manual_seed(0)

    def get_likelihoods(self, X, Y, Beta, eps=1e-06):
        inv_det = Beta.prod(dim=1)
        if (inv_det < eps).any():
            inv_det += (inv_det < eps).type(torch.FloatTensor) * eps
        det = 1 / inv_det
        None
        norm_term = 1 / torch.sqrt((2 * np.pi) ** self.dim * torch.abs(det))
        inv_covars = Beta
        dist = (Y - X).pow(2)
        exponent = (-0.5 * dist * inv_covars).sum(dim=1)
        pk = norm_term * exponent.exp()
        return pk

    def get_log_likelihoods(self, X, Y, sq_Beta, eps=1e-06):
        Beta = sq_Beta ** 2
        log_det_term = 0.5 * Beta.log().sum(dim=1)
        norm_term = -0.5 * np.log(2 * np.pi) * self.dim
        inv_covars = Beta
        dist = (Y - X).pow(2)
        exponent = (-0.5 * dist * inv_covars).sum(dim=1)
        log_p = log_det_term + exponent + norm_term
        return log_p

    def forward(self, x, y, Beta):
        p = self.get_log_likelihoods(x, y, Beta)
        E = torch.sum(-p) / x.shape[0]
        return E


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
