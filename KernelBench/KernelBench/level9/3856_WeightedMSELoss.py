import torch
from torch import nn


def assert_(condition, message='', exception_type=AssertionError):
    """Like assert, but with arbitrary exception types."""
    if not condition:
        raise exception_type(message)


class WeightedMSELoss(nn.Module):
    NEGATIVE_CLASS_WEIGHT = 1.0

    def __init__(self, positive_class_weight=1.0, positive_class_value=1.0,
        size_average=True):
        super(WeightedMSELoss, self).__init__()
        assert_(positive_class_weight >= 0,
            "Positive class weight can't be less than zero, got {}.".format
            (positive_class_weight), ValueError)
        self.mse = nn.MSELoss(size_average=size_average)
        self.positive_class_weight = positive_class_weight
        self.positive_class_value = positive_class_value

    def forward(self, input, target):
        positive_class_mask = target.data.eq(self.positive_class_value
            ).type_as(target.data)
        weight_differential = positive_class_mask.mul_(self.
            positive_class_weight - self.NEGATIVE_CLASS_WEIGHT)
        weights = weight_differential.add_(self.NEGATIVE_CLASS_WEIGHT)
        sqrt_weights = weights.sqrt_()
        return self.mse(input * sqrt_weights, target * sqrt_weights)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
