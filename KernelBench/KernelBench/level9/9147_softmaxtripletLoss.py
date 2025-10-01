import torch
import torch.nn as nn


class softmaxtripletLoss(nn.Module):

    def __init__(self):
        super(softmaxtripletLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, anchor, pos, neg):
        anchor.size(0)
        d2pos = self.dist(anchor, pos)
        d2neg = self.dist(anchor, neg)
        e_pos = torch.exp(d2pos)
        e_neg = torch.exp(d2neg)
        d_pos = e_pos / (e_pos + e_neg)
        e_neg / (e_pos + e_neg)
        loss = torch.sum(d_pos ** 2)
        return loss, (d2pos < d2neg).sum()

    def dist(self, a, b):
        d = a - b
        d = d ** 2
        d = self.relu(d)
        return torch.sqrt(torch.sum(d, dim=-1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
