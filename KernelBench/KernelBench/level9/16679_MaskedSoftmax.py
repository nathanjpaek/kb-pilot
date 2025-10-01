import torch
import torch as th
from torch import nn
import torch.nn.functional as F


class MaskedSoftmax(nn.Module):

    def __init__(self, dim):
        super(MaskedSoftmax, self).__init__()
        self.dim = dim

    def forward(self, logit, mask=None):
        if mask is None:
            dist = F.softmax(logit - th.max(logit, dim=self.dim, keepdim=
                True)[0], dim=self.dim)
        else:
            dist_ = F.softmax(logit - th.max(logit, dim=self.dim, keepdim=
                True)[0], dim=self.dim) * mask
            normalization_factor = dist_.sum(self.dim, keepdim=True)
            dist = dist_ / normalization_factor
        return dist


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
