import torch
from torch import nn
import torch.utils.checkpoint


class SqueezeBertLayerNorm(nn.LayerNorm):
    """
    This is a nn.LayerNorm subclass that accepts NCW data layout and performs normalization in the C dimension.

    N = batch C = channels W = sequence length
    """

    def __init__(self, hidden_size, eps=1e-12):
        nn.LayerNorm.__init__(self, normalized_shape=hidden_size, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = nn.LayerNorm.forward(self, x)
        return x.permute(0, 2, 1)


class ConvDropoutLayerNorm(nn.Module):
    """
    ConvDropoutLayerNorm: Conv, Dropout, LayerNorm
    """

    def __init__(self, cin, cout, groups, dropout_prob):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=cin, out_channels=cout,
            kernel_size=1, groups=groups)
        self.layernorm = SqueezeBertLayerNorm(cout)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        x = self.conv1d(hidden_states)
        x = self.dropout(x)
        x = x + input_tensor
        x = self.layernorm(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'cin': 4, 'cout': 4, 'groups': 1, 'dropout_prob': 0.5}]
