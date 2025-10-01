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


class RepulsionLoss(nn.Module):

    def __init__(self, knn=4, h=0.03):
        super().__init__()
        self.knn = knn
        self.h = h

    def forward(self, pc):
        _knn_idx, knn_dist = get_knn_idx_dist(pc, pc, k=self.knn, offset=1)
        weight = torch.exp(-knn_dist / self.h ** 2)
        loss = torch.sum(-knn_dist * weight)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
