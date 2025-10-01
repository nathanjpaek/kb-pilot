import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, dim, num_clusters=64):
        """
        Args:
            dim : int
                Dimension of descriptors
            num_clusters : int
                The number of clusters
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False
            )
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        clsts_assign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clsts_assign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]
        alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        self.conv.weight = nn.Parameter(torch.from_numpy(alpha *
            clsts_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x, crm=None):
        N, C = x.shape[:2]
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        if crm is not None:
            assert crm.shape[0] == N and crm.shape[1] == 1 and crm.shape[2:
                ] == x.shape[2:]
            soft_assign = torch.mul(soft_assign, crm.view(N, 1, -1))
        x_flatten = x.view(N, C, -1)
        vlad = torch.zeros((N, self.num_clusters, C), dtype=x.dtype, layout
            =x.layout, device=x.device)
        for c in range(self.num_clusters):
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3
                ) - self.centroids[c:c + 1, :].expand(x_flatten.size(-1), -
                1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, c:c + 1, :].unsqueeze(2)
            vlad[:, c:c + 1, :] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(N, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
