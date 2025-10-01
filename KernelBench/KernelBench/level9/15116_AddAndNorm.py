import torch
import torch.nn as nn


class AddAndNorm(nn.Module):

    def __init__(self, d_model):
        super(AddAndNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, residual):
        return self.layer_norm(x + residual)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
