import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


class GELU(nn.Module):

    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 *
            torch.pow(x, 3))))
        return x * cdf


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
