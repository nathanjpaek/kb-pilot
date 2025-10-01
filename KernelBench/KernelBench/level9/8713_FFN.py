import torch
import torch.nn as nn


class FFN(nn.Module):
    """Feed Forward Network."""

    def __init__(self, num_features: 'int', ffn_dim_1: 'int', ffn_dim_2: 'int'
        ) ->None:
        """Initialize the class."""
        super().__init__()
        self.gemm1 = nn.Linear(num_features, ffn_dim_1, bias=False)
        self.relu = nn.ReLU()
        self.gemm2 = nn.Linear(ffn_dim_1, ffn_dim_2, bias=False)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Step forward."""
        out = self.gemm1(x)
        out = self.relu(out)
        out = self.gemm2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4, 'ffn_dim_1': 4, 'ffn_dim_2': 4}]
