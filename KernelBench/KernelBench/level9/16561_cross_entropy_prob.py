import torch
import torch.nn as nn
import torch.nn.functional as F


class cross_entropy_prob(nn.Module):

    def __init__(self):
        super(cross_entropy_prob, self).__init__()

    def forward(self, pred, soft_targets):
        pred = F.log_softmax(pred)
        loss = torch.mean(torch.sum(-soft_targets * pred, 1))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
