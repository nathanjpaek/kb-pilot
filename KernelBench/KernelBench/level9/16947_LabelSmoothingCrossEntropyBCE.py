import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropyBCE(nn.Module):

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyBCE, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        target = target.float() * self.confidence + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(x, target.type_as(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
