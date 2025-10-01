import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F
import torch.utils.data


def hard_examples_mining(dist_mat, identity_mat, return_idxes=False):
    """Select hard positives and hard negatives according to `In defense of the Triplet Loss for Person
    Re-Identification (ICCV 2017) <https://arxiv.org/pdf/1703.07737v2.pdf>`_

    Args:
        dist_mat (tensor): pairwise distance matrix between two sets of features
        identity_mat (tensor): a matrix of shape :math:`(N, M)`. If two images :math:`P[i]` of set :math:`P` and
            :math:`Q[j]` of set :math:`Q` come from the same person, then :math:`identity\\_mat[i, j] = 1`,
            otherwise :math:`identity\\_mat[i, j] = 0`
        return_idxes (bool, optional): if True, also return indexes of hard examples. Default: False
    """
    sorted_dist_mat, sorted_idxes = torch.sort(dist_mat + -10000000.0 * (1 -
        identity_mat), dim=1, descending=True)
    dist_ap = sorted_dist_mat[:, 0]
    hard_positive_idxes = sorted_idxes[:, 0]
    sorted_dist_mat, sorted_idxes = torch.sort(dist_mat + 10000000.0 *
        identity_mat, dim=1, descending=False)
    dist_an = sorted_dist_mat[:, 0]
    hard_negative_idxes = sorted_idxes[:, 0]
    if return_idxes:
        return dist_ap, dist_an, hard_positive_idxes, hard_negative_idxes
    return dist_ap, dist_an


def pairwise_euclidean_distance(x, y):
    """Compute pairwise euclidean distance between two sets of features"""
    m, n = x.size(0), y.size(0)
    dist_mat = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n) + torch.pow(y,
        2).sum(1, keepdim=True).expand(n, m).t() - 2 * torch.matmul(x, y.t())
    dist_mat = dist_mat.clamp(min=1e-12).sqrt()
    return dist_mat


class TripletLoss(nn.Module):
    """Triplet loss augmented with batch hard from `In defense of the Triplet Loss for Person Re-Identification
    (ICCV 2017) <https://arxiv.org/pdf/1703.07737v2.pdf>`_.

    Args:
        margin (float): margin of triplet loss
        normalize_feature (bool, optional): if True, normalize features into unit norm first before computing loss.
            Default: False.
    """

    def __init__(self, margin, normalize_feature=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, f, labels):
        if self.normalize_feature:
            f = F.normalize(f)
        dist_mat = pairwise_euclidean_distance(f, f)
        n = dist_mat.size(0)
        identity_mat = labels.expand(n, n).eq(labels.expand(n, n).t()).float()
        dist_ap, dist_an = hard_examples_mining(dist_mat, identity_mat)
        y = torch.ones_like(dist_ap)
        loss = self.margin_loss(dist_an, dist_ap, y)
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'margin': 4}]
