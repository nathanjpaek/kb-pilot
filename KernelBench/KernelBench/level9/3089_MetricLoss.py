import torch
import torch.nn as nn
import torch.jit
import torch.nn


class MetricLoss(nn.Module):
    """Loss designed to train a true metric, as opposed to a
  sigmoid classifier.
  """

    def __init__(self):
        super(MetricLoss, self).__init__()

    def forward(self, input, target):
        weight = 1.0 - target
        weight /= weight.sum()
        weight += target / target.sum()
        tensor_result = weight * (input - target) ** 2
        return tensor_result.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
