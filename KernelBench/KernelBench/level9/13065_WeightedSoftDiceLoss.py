import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn
import torch.utils.data


class WeightedSoftDiceLoss(nn.Module):

    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        w = weights.view(num, -1)
        w2 = w * w
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = m1 * m2
        score = 2.0 * ((w2 * intersection).sum(1) + 1) / ((w2 * m1).sum(1) +
            (w2 * m2).sum(1) + 1)
        score = 1 - score.sum() / num
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
