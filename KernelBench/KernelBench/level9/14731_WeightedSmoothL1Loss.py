import torch
import torch.nn as nn


class WeightedSmoothL1Loss(nn.SmoothL1Loss):

    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction='none')
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)
        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold
        l1[mask] = l1[mask] * self.weight
        return l1.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'threshold': 4, 'initial_weight': 4}]
