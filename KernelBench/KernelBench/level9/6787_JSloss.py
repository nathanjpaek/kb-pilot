import torch
import torch.nn as nn
import torch.nn.functional as F


class JSloss(nn.Module):
    """  Compute the Jensen-Shannon loss using the torch native kl_div"""

    def __init__(self, reduction='batchmean'):
        super().__init__()
        self.red = reduction

    def forward(self, input, target):
        net = (input + target) / 2.0
        return 0.5 * (F.kl_div(input, net, reduction=self.red) + F.kl_div(
            target, net, reduction=self.red))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
