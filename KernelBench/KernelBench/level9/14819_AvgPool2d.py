import torch
import torch.nn as nn
import torch.nn.functional as F


def keep_variance_fn(x):
    return x + 0.001


class AvgPool2d(nn.Module):

    def __init__(self, keep_variance_fn=None, kernel_size=2):
        super(AvgPool2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.kernel_size = kernel_size

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.avg_pool2d(inputs_mean, self.kernel_size, stride=2,
            padding=1)
        outputs_variance = F.avg_pool2d(inputs_variance, self.kernel_size,
            stride=2, padding=1)
        outputs_variance = outputs_variance / (inputs_mean.size(2) *
            inputs_mean.size(3))
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance / (inputs_mean.shape[2] *
            inputs_mean.shape[3])


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
