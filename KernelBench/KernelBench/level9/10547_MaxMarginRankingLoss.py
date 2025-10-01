import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=1):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, x):
        n = x.size()[0]
        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)
        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)
        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))
        return max_margin.mean()


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
