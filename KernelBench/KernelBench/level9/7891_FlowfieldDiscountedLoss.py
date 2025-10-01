import torch
import torch.nn as nn
import torch.utils.data
import torch.random
import torch.nn.functional as F


class FlowfieldDiscountedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, source, target, flowfield):
        loss = F.mse_loss(source, target, reduction='none')
        bad_bits = flowfield ** 2
        bad_bits[bad_bits <= 1.0] = 1.0
        bad_bits[bad_bits > 1.0] = 0
        mask = torch.prod(bad_bits, 3).expand(1, -1, -1, -1).permute(1, 0, 2, 3
            )
        loss = loss * mask
        return torch.sum(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
