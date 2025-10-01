import torch
from torch import nn
import torch.onnx


class PRelu(nn.Module):

    def forward(self, X: 'torch.Tensor', slope: 'torch.Tensor'):
        return torch.clamp(X, min=0) + torch.clamp(X, max=0) * slope


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
