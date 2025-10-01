import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import torch.optim
from torch.distributions import Normal


class homo_Gauss_mloglike(nn.Module):

    def __init__(self, Ndims=1, sig=None):
        super(homo_Gauss_mloglike, self).__init__()
        if sig is None:
            self.log_std = nn.Parameter(torch.zeros(Ndims))
        else:
            self.log_std = nn.Parameter(torch.ones(Ndims) * np.log(sig),
                requires_grad=False)

    def forward(self, mu, y, model_std=None):
        sig = self.log_std.exp().clamp(min=0.0001)
        if model_std is not None:
            sig = (sig ** 2 + model_std ** 2) ** 0.5
        dist = Normal(mu, sig)
        return -dist.log_prob(y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
