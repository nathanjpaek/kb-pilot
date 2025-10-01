import torch
import torch.nn as nn


class WeightL1Loss(nn.Module):

    def __init__(self):
        super(WeightL1Loss, self).__init__()

    def forward(self, pred_loc, label_loc, loss_weight):
        b, _, sh, sw = pred_loc.size()
        pred_loc = pred_loc.view(b, 4, -1, sh, sw)
        diff = (pred_loc - label_loc).abs()
        diff = diff.sum(dim=1).view(b, -1, sh, sw)
        loss = diff * loss_weight
        return loss.sum().div(b)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
