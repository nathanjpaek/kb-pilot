import torch
import torch.nn as nn


class DSCLoss(nn.Module):

    def __init__(self):
        super(DSCLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        input_flat * target_flat
        numerator = 2 * ((1 - input_flat) * input_flat * target_flat).sum(1
            ) + smooth
        denominator = ((1 - input_flat) * input_flat + target_flat).sum(1
            ) + smooth
        loss = 1 - numerator / denominator
        return loss.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
