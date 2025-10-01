import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):

    def __init__(self, dims, num_clusters, outdims=None):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dims = dims
        self.centroids = nn.Parameter(torch.randn(num_clusters, dims) /
            math.sqrt(self.dims))
        self.conv = nn.Conv2d(dims, num_clusters, kernel_size=1, bias=False)
        if outdims is not None:
            self.outdims = outdims
            self.reduction_layer = nn.Linear(self.num_clusters * self.dims,
                self.outdims, bias=False)
        else:
            self.outdims = self.num_clusters * self.dims
        self.norm = nn.LayerNorm(self.outdims)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.weight = nn.Parameter(self.centroids.detach().clone().
            unsqueeze(-1).unsqueeze(-1))
        if hasattr(self, 'reduction_layer'):
            nn.init.normal_(self.reduction_layer.weight, std=1 / math.sqrt(
                self.num_clusters * self.dims))

    def forward(self, x, mask=None, sample=False):
        N, C, T, R = x.shape
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1).view(N, self.
            num_clusters, T, R)
        x_flatten = x.view(N, C, -1)
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout
            =x.layout, device=x.device)
        for cluster in range(self.num_clusters):
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3
                ) - self.centroids[cluster:cluster + 1, :].expand(x_flatten
                .size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual.view(N, C, T, R)
            residual *= soft_assign[:, cluster:cluster + 1, :]
            if mask is not None:
                residual = residual.masked_fill((1 - mask.unsqueeze(1).
                    unsqueeze(-1)).bool(), 0.0)
            vlad[:, cluster:cluster + 1, :] = residual.sum([-2, -1]).unsqueeze(
                1)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        if hasattr(self, 'reduction_layer'):
            vlad = self.reduction_layer(vlad)
        return self.norm(vlad)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dims': 4, 'num_clusters': 4}]
