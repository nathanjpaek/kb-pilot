import torch
from torch import nn


class affinity_loss(nn.Module):

    def __init__(self):
        super(affinity_loss, self).__init__()

    def forward(self, pixel_affinity, sal_affinity, sal_diff):
        loss = torch.mean(pixel_affinity * (1 - sal_affinity)
            ) + 4 * torch.mean(sal_diff * sal_affinity)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
