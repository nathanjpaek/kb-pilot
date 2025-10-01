import torch
import torch.nn as nn
import torch.nn.functional as F


class JS_div(nn.Module):

    def __init__(self, margin=0.1):
        super(JS_div, self).__init__()
        self.margin = margin
        self.dist = nn.CosineSimilarity(dim=0)
        self.KLDivloss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, feat1, feat2, get_softmax=True):
        if get_softmax:
            feat11 = F.softmax(feat1)
            feat22 = F.softmax(feat2)
        log_mean_output = ((feat11 + feat22) / 2).log()
        dis_final = (self.KLDivloss(log_mean_output, feat11) + self.
            KLDivloss(log_mean_output, feat22)) / 2
        return dis_final


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
