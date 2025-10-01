import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class LogitBinaryCrossEntropy(nn.Module):

    def __init__(self):
        super(LogitBinaryCrossEntropy, self).__init__()

    def forward(self, pred_score, target_score, weights=None):
        loss = F.binary_cross_entropy_with_logits(pred_score, target_score,
            size_average=True)
        loss = loss * target_score.size(1)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
