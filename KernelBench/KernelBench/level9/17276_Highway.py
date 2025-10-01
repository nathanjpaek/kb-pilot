import torch
import torch.nn as nn
import torch.nn.utils


class Highway(nn.Module):

    def __init__(self, conv_out_dim, e_word):
        super().__init__()
        self.conv_out_dim = conv_out_dim
        self.e_word = e_word
        self.linear_proj = nn.Linear(conv_out_dim, self.e_word)
        self.linear_gate = nn.Linear(self.conv_out_dim, self.e_word)

    def forward(self, x_conv_out):
        x_proj = nn.functional.relu(self.linear_proj(x_conv_out))
        x_gate = self.linear_gate(x_conv_out)
        x_highway = x_proj * x_gate + (1.0 - x_gate) * x_conv_out
        return x_highway


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'conv_out_dim': 4, 'e_word': 4}]
