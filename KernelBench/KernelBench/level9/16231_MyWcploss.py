import torch
from torch import nn


class MyWcploss(nn.Module):

    def __init__(self):
        super(MyWcploss, self).__init__()

    def forward(self, pred, gt):
        eposion = 1e-10
        torch.sigmoid(pred)
        count_pos = torch.sum(gt) * 1.0 + eposion
        count_neg = torch.sum(1.0 - gt) * 1.0
        beta = count_neg / count_pos
        beta_back = count_pos / (count_pos + count_neg)
        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back * bce1(pred, gt)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
