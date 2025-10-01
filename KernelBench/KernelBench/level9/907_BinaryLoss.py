import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLoss(nn.Module):

    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]
        loss = (pos_loss.sum() + neg_loss.sum()) / (pos_loss.size(0) +
            neg_loss.size(0))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
