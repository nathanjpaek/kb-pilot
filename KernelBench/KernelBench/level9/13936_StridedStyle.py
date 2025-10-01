import torch
import torch.nn as nn


class NamedTensor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class StridedStyle(nn.ModuleList):

    def __init__(self, n_latents):
        super().__init__([NamedTensor() for _ in range(n_latents)])
        self.n_latents = n_latents

    def forward(self, x):
        styles = [self[i](x[:, i, :]) for i in range(self.n_latents)]
        return torch.stack(styles, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_latents': 4}]
