import torch
import torch.nn as nn
import torch.nn.utils


class Highway(nn.Module):
    """it is not fun"""

    def __init__(self, e_word_size, drop_rate=0.3):
        super(Highway, self).__init__()
        self.w_proj = nn.Linear(e_word_size, e_word_size)
        self.w_gate = nn.Linear(e_word_size, e_word_size)
        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x_conv_out):
        """
        Map from x_conv_out to x_highway with batches
        @para x_conv_out: shape (b, e_word_size): b - batch size, e_word_size
        @return x_highway: shape (b, e_word_size)
        """
        x_proj = self.relu(self.w_proj(x_conv_out))
        x_gate = self.sigmoid(self.w_gate(x_proj))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return self.dropout(x_highway)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'e_word_size': 4}]
