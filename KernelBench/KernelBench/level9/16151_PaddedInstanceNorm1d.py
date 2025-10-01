import torch
import torch.nn as nn


class PaddedInstanceNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False,
        track_running_stats=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        if affine is True:
            raise NotImplementedError
        if track_running_stats is True:
            raise NotImplementedError

    def forward(self, x, lengths):
        lengths = lengths.view(-1, 1, 1).float()
        sum_ = torch.sum(x, dim=2, keepdim=True)
        mean = sum_ / lengths
        sqsum = torch.sum(torch.pow(x, 2.0), dim=2, keepdim=True)
        sqmean = sqsum / lengths
        var = sqmean - torch.pow(mean, 2.0)
        return (x - mean) / torch.pow(var + self.eps, 0.5)


def get_inputs():
    return [torch.rand([4, 16384, 4, 4]), torch.rand([4, 256, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
