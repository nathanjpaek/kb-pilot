import torch
import torch.nn as nn
import torch.nn.functional as F


class BPR_max(nn.Module):

    def __init__(self):
        super(BPR_max, self).__init__()

    def forward(self, logit):
        logit_softmax = F.softmax(logit, dim=1)
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        loss = -torch.log(torch.mean(logit_softmax * torch.sigmoid(diff)))
        return loss


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
