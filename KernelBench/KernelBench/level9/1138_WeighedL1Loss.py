import torch
from torch import Tensor
from torch.nn import L1Loss


class WeighedL1Loss(L1Loss):

    def __init__(self, weights):
        super().__init__(reduction='none')
        self.weights = weights

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        loss = super().forward(input, target)
        return (loss * self.weights).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'weights': 4}]
