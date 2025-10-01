import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.autograd


class WeightedL2WithSigmaLoss(nn.Module):

    def __init__(self, code_weights: 'list'=None):
        super(WeightedL2WithSigmaLoss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights)
        else:
            self.code_weights = None

    @staticmethod
    def l2_loss(diff, sigma=None):
        if sigma is None:
            loss = 0.5 * diff ** 2
        else:
            loss = 0.5 * (diff / torch.exp(sigma)) ** 2 + math.log(math.
                sqrt(6.28)) + sigma
        return loss

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor',
        weights: 'torch.Tensor'=None, sigma: 'torch.Tensor'=None):
        target = torch.where(torch.isnan(target), input, target)
        diff = input - target
        if self.code_weights is not None:
            diff = diff * self.code_weights
        loss = self.l2_loss(diff, sigma=sigma)
        if weights is not None:
            assert weights.shape == loss.shape[:-1]
            weights = weights.unsqueeze(-1)
            assert len(loss.shape) == len(weights.shape)
            loss = loss * weights
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
