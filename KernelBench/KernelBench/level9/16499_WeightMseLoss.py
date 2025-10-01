import torch
import torch.nn as nn


class WeightMseLoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightMseLoss, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, targets, weights):
        """ inputs is N * C
            targets is N * C
            weights is N * C
        """
        N = inputs.size(0)
        C = inputs.size(1)
        out = targets - inputs
        out = weights * torch.pow(out, 2)
        loss = out.sum()
        if self.size_average:
            loss = loss / (N * C)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
