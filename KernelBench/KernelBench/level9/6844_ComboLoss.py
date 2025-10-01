import torch
import torch.nn as nn


class ComboLoss(nn.Module):

    def __init__(self, weight=None, size_average=True, alpha=0.5, ce_ratio=0.5
        ):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio

    def forward(self, inputs, targets, smooth=1):
        e = 1e-07
        inputs = torch.sigmoid(inputs)
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum(
            ) + smooth)
        inputs = torch.clamp(inputs, e, 1.0 - e)
        out = -(self.alpha * (targets * torch.log(inputs)) + (1 - self.
            alpha) * (1.0 - targets) * torch.log(1.0 - inputs))
        weighted_ce = out.mean(-1)
        combo = self.ce_ratio * weighted_ce - (1 - self.ce_ratio) * dice
        return combo


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
