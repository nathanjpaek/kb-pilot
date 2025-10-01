import torch
import numpy as np
import torch.nn as nn
from numbers import Number


def keep_variance_fn(x):
    return x + 0.001


def normcdf(value, mu=0.0, stddev=1.0):
    sinv = 1.0 / stddev if isinstance(stddev, Number) else stddev.reciprocal()
    return 0.5 * (1.0 + torch.erf((value - mu) * sinv / np.sqrt(2.0)))


def _normal_log_pdf(value, mu, stddev):
    var = stddev ** 2
    log_scale = np.log(stddev) if isinstance(stddev, Number) else torch.log(
        stddev)
    return -(value - mu) ** 2 / (2.0 * var) - log_scale - np.log(np.sqrt(
        2.0 * np.pi))


def normpdf(value, mu=0.0, stddev=1.0):
    return torch.exp(_normal_log_pdf(value, mu, stddev))


class ReLU(nn.Module):

    def __init__(self, keep_variance_fn=None):
        super(ReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance):
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        outputs_mean = features_mean * cdf + features_stddev * pdf
        outputs_variance = (features_mean ** 2 + features_variance
            ) * cdf + features_mean * features_stddev * pdf - outputs_mean ** 2
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
