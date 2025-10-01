import torch
from torch.nn.modules.loss import MSELoss


class HalfMSELoss(MSELoss):

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)

    def forward(self, input, target):
        return super().forward(input, target) / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
