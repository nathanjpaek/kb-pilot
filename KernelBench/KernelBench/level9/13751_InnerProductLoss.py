import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.autograd


class InnerProductLoss(nn.Module):

    def __init__(self, code_weights: 'list'=None):
        super(InnerProductLoss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights)
        else:
            self.code_weights = None

    @staticmethod
    def ip_loss(product):
        return 1 - product.mean(dim=-1, keepdim=True)

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor',
        weights: 'torch.Tensor'=None):
        target = torch.where(torch.isnan(target), input, target)
        product = input * target
        if self.code_weights is not None:
            product = product * self.code_weights
        loss = self.ip_loss(product)
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
