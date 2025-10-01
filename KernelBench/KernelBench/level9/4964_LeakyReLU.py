import torch
import numpy as np
import torch.nn as nn
from numbers import Number


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


def keep_variance_fn(x):
    return x + 0.001


class LeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.01, keep_variance_fn=None):
        super(LeakyReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self._negative_slope = negative_slope

    def forward(self, features_mean, features_variance):
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        negative_cdf = 1.0 - cdf
        mu_cdf = features_mean * cdf
        stddev_pdf = features_stddev * pdf
        squared_mean_variance = features_mean ** 2 + features_variance
        mean_stddev_pdf = features_mean * stddev_pdf
        mean_r = mu_cdf + stddev_pdf
        variance_r = (squared_mean_variance * cdf + mean_stddev_pdf - 
            mean_r ** 2)
        mean_n = -features_mean * negative_cdf + stddev_pdf
        variance_n = (squared_mean_variance * negative_cdf -
            mean_stddev_pdf - mean_n ** 2)
        covxy = -mean_r * mean_n
        outputs_mean = mean_r - self._negative_slope * mean_n
        outputs_variance = (variance_r + self._negative_slope * self.
            _negative_slope * variance_n - 2.0 * self._negative_slope * covxy)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
