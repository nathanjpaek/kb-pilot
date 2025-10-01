import torch
from torch import nn
import torch.onnx


class ThresholdedRelu(nn.Module):

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__()

    def forward(self, X: 'torch.Tensor'):
        Y = torch.clamp(X, min=self.alpha)
        Y[Y == self.alpha] = 0.0
        return Y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
