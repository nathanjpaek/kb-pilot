import math
import torch
import torch.nn as nn


def get_knn_idx_dist(pos: 'torch.FloatTensor', query: 'torch.FloatTensor',
    k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    B, N, F = tuple(pos.size())
    M = query.size(1)
    pos = pos.unsqueeze(1).expand(B, M, N, F)
    query = query.unsqueeze(2).expand(B, M, N, F)
    dist = torch.sum((pos - query) ** 2, dim=3, keepdim=False)
    knn_idx = torch.argsort(dist, dim=2)[:, :, offset:k + offset]
    knn_dist = torch.gather(dist, dim=2, index=knn_idx)
    return knn_idx, knn_dist


def group(x: 'torch.FloatTensor', idx: 'torch.LongTensor'):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())
    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)
    return torch.gather(x, dim=2, index=idx)


class ProjectionLoss(nn.Module):

    def __init__(self, knn=8, sigma_p=0.03, sigma_n=math.radians(15)):
        super().__init__()
        self.sigma_p = sigma_p
        self.sigma_n = sigma_n
        self.knn = knn

    def distance_weight(self, dist):
        """
        :param  dist: (B, N, k), Squared L2 distance
        :return (B, N, k)
        """
        return torch.exp(-dist / self.sigma_p ** 2)

    def angle_weight(self, nb_normals):
        """
        :param  nb_normals: (B, N, k, 3), Normals of neighboring points 
        :return (B, N, k)
        """
        estm_normal = nb_normals[:, :, 0:1, :]
        inner_prod = (nb_normals * estm_normal.expand_as(nb_normals)).sum(dim
            =-1)
        return torch.exp(-(1 - inner_prod) / (1 - math.cos(self.sigma_n)))

    def forward(self, preds, gts, normals, **kwargs):
        knn_idx, knn_dist = get_knn_idx_dist(gts, query=preds, k=self.knn,
            offset=0)
        nb_points = group(gts, idx=knn_idx)
        nb_normals = group(normals, idx=knn_idx)
        distance_w = self.distance_weight(knn_dist)
        angle_w = self.angle_weight(nb_normals)
        weights = distance_w * angle_w
        inner_prod = ((preds.unsqueeze(-2).expand_as(nb_points) - nb_points
            ) * nb_normals).sum(dim=-1)
        inner_prod = torch.abs(inner_prod)
        point_displacement = (inner_prod * weights).sum(dim=-1) / weights.sum(
            dim=-1)
        return point_displacement.sum()


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {}]
