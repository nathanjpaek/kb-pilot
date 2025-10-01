import torch
from torch import nn
import torch.nn.functional as F


class TripletMarginLossCosine(nn.Module):

    def __init__(self, margin=1.0):
        super(TripletMarginLossCosine, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_p = 1 - F.cosine_similarity(anchor, positive).view(-1, 1)
        d_n = 1 - F.cosine_similarity(anchor, negative).view(-1, 1)
        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
