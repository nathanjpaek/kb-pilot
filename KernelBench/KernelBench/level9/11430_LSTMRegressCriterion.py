import torch
import torch.nn as nn


class LSTMRegressCriterion(nn.Module):

    def __init__(self):
        super(LSTMRegressCriterion, self).__init__()

    def forward(self, pred, target, mask):
        pred = pred.clone()
        target = target.clone()
        mask = mask.clone()
        target = target[:, :pred.size(1), :]
        mask = mask[:, :pred.size(1), :]
        diff = 0.5 * (pred - target) ** 2
        diff = diff * mask
        output = torch.sum(diff) / torch.sum(mask)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
