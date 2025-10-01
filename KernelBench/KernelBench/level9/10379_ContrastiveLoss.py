import torch
import torch.nn as nn
import torch.nn.init
import torch.utils.data
import torch.utils.data.distributed


def cosine_sim(im, s):
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

    def forward(self, hs):
        hs.size(1)
        tgt = hs[:, -1, :]
        src = hs[:, 0:-1, :]
        tgt = tgt.unsqueeze(1)
        scores = (tgt * src).sum(dim=2)
        d = scores[:, -1]
        d = d.unsqueeze(1)
        scores = scores[:, 0:-1]
        d = d.expand_as(scores)
        cost = (self.margin + scores - d).clamp(min=0)
        return cost.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
