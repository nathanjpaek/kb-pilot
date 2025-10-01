import torch
import torch.nn as nn
from torch.autograd import *
import torch.nn.init


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1)
        ) - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class knn_ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False):
        super(knn_ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

    def forward(self, im, knn_im, s):
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        knn_scores = self.sim(knn_im, s)
        knn_diagonal = knn_scores.diag().view(knn_im.size(0), 1)
        cost = (self.margin + knn_diagonal - diagonal).clamp(min=0)
        return cost


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
