import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    contrastive loss
    L2 distance:
    L(a1,a2,y) = y * d(a1, a2) + (1-y)*max(0, m - d(a1, a2))
    cosine distance:
    L(a1, a2, y) = y * (1 - d(a1,a2)) + (1-y) * max(0, d(a1,a2) -m)

    where y=1 if (a1,a2) relevant else 0
    """

    def __init__(self, margin=1.0, metric='l2'):
        super().__init__()
        self.margin = margin
        self.metric = metric
        metric_list = ['l2', 'cosine']
        assert metric in metric_list, 'Error! contrastive metric %s not supported.' % metric
        self.metric_id = metric_list.index(metric)

    def forward(self, x, y):
        a, p = x.chunk(2, dim=0)
        if self.metric_id == 0:
            dist = torch.sum((a - p) ** 2, dim=1)
            loss = y * dist + (1 - y) * F.relu(self.margin - dist)
        else:
            dist = F.cosine_similarity(a, p)
            loss = y * (1 - dist) + (1 - y) * F.relu(dist - self.margin)
        return loss.mean() / 2.0

    def extra_repr(self) ->str:
        return '?xD -> scalar (Loss)'


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 2, 4, 4])]


def get_init_inputs():
    return [[], {}]
