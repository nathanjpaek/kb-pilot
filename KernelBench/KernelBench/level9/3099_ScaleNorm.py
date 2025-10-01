import torch
import torch.nn as nn
import torch.jit
import torch.nn


class ScaleNorm(nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float))

    def forward(self, inputs):
        out = inputs.view(inputs.size(0), -1)
        norm = out.norm(dim=1, keepdim=True)
        out = self.scale * out / (norm + 1e-16)
        return out.view(*inputs.shape)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
