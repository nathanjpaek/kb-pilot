import torch
import torch.nn as nn


class AddAndNorm(nn.Module):

    def __init__(self, d_model, p_drop):
        super(AddAndNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, inputs, x):
        return self.layer_norm(inputs + self.dropout(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'p_drop': 0.5}]
