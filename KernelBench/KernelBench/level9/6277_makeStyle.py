import torch
import torch.nn as nn
import torch.nn.functional as F


class makeStyle(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x0):
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style ** 2, axis=1, keepdim=True) ** 0.5
        return style


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
