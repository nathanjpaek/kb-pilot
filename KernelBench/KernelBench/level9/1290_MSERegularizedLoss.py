from torch.nn import Module
import torch
import torch as tc
import torch.nn.functional as F
from torch.nn.modules.module import Module


class MSERegularizedLoss(Module):

    def __init__(self, alpha=1):
        super(MSERegularizedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, weights, prediction, target):
        mse = F.mse_loss(prediction, target)
        reg = tc.sum(tc.pow(weights, 2))
        return mse + self.alpha * reg


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
