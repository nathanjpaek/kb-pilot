import torch
import torch as th
import torch.nn as nn


class BinaryFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target, weight=None):
        if weight is not None:
            assert weight.size() == input.size(
                ), f'weight size: {weight.size()}, input size: {input.size()}'
            assert (weight >= 0).all() and (weight <= 1).all(
                ), f'weight max: {weight.max()}, min: {weight.min()}'
        input = input.clamp(1e-06, 1.0 - 1e-06)
        if weight is None:
            loss = th.sum(-self.alpha * target * (1 - input) ** self.gamma *
                th.log(input) - (1 - self.alpha) * (1 - target) * input **
                self.gamma * th.log(1 - input))
        else:
            loss = th.sum((-self.alpha * target * (1 - input) ** self.gamma *
                th.log(input) - (1 - self.alpha) * (1 - target) * input **
                self.gamma * th.log(1 - input)) * weight)
        if self.size_average:
            loss /= input.nelement()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
