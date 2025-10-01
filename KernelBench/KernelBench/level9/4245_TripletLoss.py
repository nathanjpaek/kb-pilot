import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


def euclidean_dist(x: 'Tensor', y: 'Tensor') ->Tensor:
    xx, yy = torch.meshgrid((x ** 2).sum(1), (y ** 2).sum(1))
    return xx + yy - 2 * (x @ y.t())


class TripletLoss(nn.Module):
    """
    Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'
    """

    def __init__(self, margin: 'float'=0.3) ->None:
        super().__init__()
        self.margin = margin

    def forward(self, features: 'Tensor', targets: 'Tensor') ->Tensor:
        dist = euclidean_dist(features, features)
        is_pos = targets.unsqueeze(0) == targets.unsqueeze(1)
        is_neg = ~is_pos
        dist_ap, dist_an = self.hard_example_mining(dist, is_pos, is_neg)
        y = torch.ones_like(dist_an)
        if self.margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, self.margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            if loss == float('inf'):
                loss = F.margin_ranking_loss(dist_an, dist_ap, y, 0.3)
        return loss

    def hard_example_mining(self, dist, is_pos, is_neg):
        """For each anchor, find the hardest positive and negative sample.
        Args:
            dist_mat: pair wise distance between samples, shape [N, M]
            is_pos: positive index with shape [N, M]
            is_neg: negative index with shape [N, M]
        Returns:
            dist_ap: Tensor, distance(anchor, positive); shape [N]
            dist_an: Tensor, distance(anchor, negative); shape [N]
        NOTE: Only consider the case in which all labels have same num of samples,
        thus we can cope with all anchors in parallel.
        """
        assert len(dist.size()) == 2
        dist_ap = torch.max(dist * is_pos, dim=1)[0]
        dist_an = torch.min(dist * is_neg + is_pos * 1000000000.0, dim=1)[0]
        return dist_ap, dist_an

    def weighted_example_mining(self, dist, is_pos, is_neg):
        """For each anchor, find the weighted positive and negative sample.
        """
        assert len(dist.size()) == 2
        dist_ap = dist * is_pos
        dist_an = dist * is_neg
        weights_ap = self.softmax_weights(dist_ap, is_pos)
        weights_an = self.softmax_weights(-dist_an, is_neg)
        dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
        dist_an = torch.sum(dist_an * weights_an, dim=1)
        return dist_ap, dist_an

    def softmax_weights(self, dist, mask):
        max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
        diff = dist - max_v
        Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-06
        W = torch.exp(diff) * mask / Z
        return W


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
