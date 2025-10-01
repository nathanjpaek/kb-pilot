import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):

    def forward(self, pos_score, neg_score, average=True):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]
        loss = pos_loss.sum() + neg_loss.sum()
        if average:
            loss /= pos_loss.size(0) + neg_loss.size(0)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
