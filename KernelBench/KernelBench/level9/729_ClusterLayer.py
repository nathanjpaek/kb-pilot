import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.parameter import Parameter


class ClusterLayer(nn.Module):

    def __init__(self, n_cluster, expansion, cluster_m):
        super(ClusterLayer, self).__init__()
        self.center = Parameter(torch.Tensor(n_cluster, expansion))
        self.m = cluster_m

    def forward(self, x):
        mu = 1.0 / torch.sum(torch.abs(x.unsqueeze(1) - self.center) ** (
            2.0 / (self.m - 1.0)), dim=2)
        mu = mu / torch.sum(mu, dim=1, keepdim=True)
        return mu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_cluster': 4, 'expansion': 4, 'cluster_m': 4}]
