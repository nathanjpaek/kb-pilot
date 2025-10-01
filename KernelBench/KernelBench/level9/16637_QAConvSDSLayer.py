import torch
import torch.nn as nn


class QAConvSDSLayer(nn.Module):
    """Conv SDS layer for qa output"""

    def __init__(self, input_size: 'int', hidden_dim: 'int'):
        """
        Args:
            input_size (int): max sequence lengths
            hidden_dim (int): backbones's hidden dimension
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=
            input_size * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=input_size * 2, out_channels=
            input_size, kernel_size=1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = x + torch.relu(out)
        out = self.layer_norm(out)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_dim': 4}]
