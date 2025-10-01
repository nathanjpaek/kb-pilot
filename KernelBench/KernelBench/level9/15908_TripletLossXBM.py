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


class TripletLossXBM(nn.Module):
    """Triplet loss augmented with batch hard from `In defense of the Triplet Loss for Person Re-Identification
    (ICCV 2017) <https://arxiv.org/pdf/1703.07737v2.pdf>`_. The only difference from triplet loss lies in that
    both features from current mini batch and external storage (XBM) are involved.

    Args:
        margin (float, optional): margin of triplet loss. Default: 0.3
        normalize_feature (bool, optional): if True, normalize features into unit norm first before computing loss.
            Default: False

    Inputs:
        - f (tensor): features of current mini batch, :math:`f`
        - labels (tensor): identity labels for current mini batch, :math:`labels`
        - xbm_f (tensor): features collected from XBM, :math:`xbm\\_f`
        - xbm_labels (tensor): corresponding identity labels of xbm_f, :math:`xbm\\_labels`

    Shape:
        - f: :math:`(minibatch, F)`, where :math:`F` is the feature dimension
        - labels: :math:`(minibatch, )`
        - xbm_f: :math:`(minibatch, F)`
        - xbm_labels: :math:`(minibatch, )`
    """

    def __init__(self, margin=0.3, normalize_feature=False):
        super(TripletLossXBM, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, f, labels, xbm_f, xbm_labels):
        if self.normalize_feature:
            f = F.normalize(f)
            xbm_f = F.normalize(xbm_f)
        dist_mat = pairwise_euclidean_distance(f, xbm_f)
        n, m = f.size(0), xbm_f.size(0)
        identity_mat = labels.expand(m, n).t().eq(xbm_labels.expand(n, m)
            ).float()
        dist_ap, dist_an = hard_examples_mining(dist_mat, identity_mat)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
