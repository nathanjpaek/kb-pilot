import math
import torch
from torch import Tensor
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Learnable position embeddings

    Args:
        pe_type (str): type of position embeddings,
            which is chosen from ['fully_learnable', 'sinusoidal']
        d_model (int): embed dim (required).
        max_len (int): max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model, max_len=100)
    """

    def __init__(self, pe_type: 'str', d_model: 'int', max_len: 'int'=5000):
        super(PositionalEncoding, self).__init__()
        if pe_type == 'fully_learnable':
            self.pe = nn.parameter.Parameter(torch.randn(max_len, 1, d_model))
        elif pe_type == 'sinusoidal':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-
                math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
        else:
            raise RuntimeError(
                'PE type should be fully_learnable/sinusoidal, not {}'.
                format(pe_type))

    def forward(self, x: 'Tensor') ->Tensor:
        """Inputs of forward function
        Args:
            x (Tensor): the sequence fed to the positional encoder model [L, N, C]
        Returns:
            output (Tensor): position embeddings [L, N, C]
        """
        return x + self.pe[:x.size(0)]


class SimpleEncoder(nn.Module):

    def __init__(self, feature_dim, encoder_dim, image_size=14):
        super(SimpleEncoder, self).__init__()
        self.linear = nn.Conv2d(feature_dim, encoder_dim, kernel_size=1)
        self.positional_encoding = PositionalEncoding('fully_learnable', 2 *
            encoder_dim, image_size ** 2)

    def forward(self, x1, x2):
        return self.positional_encoding(torch.cat([self.linear(x1).flatten(
            start_dim=2).permute(2, 0, 1), self.linear(x2).flatten(
            start_dim=2).permute(2, 0, 1)], dim=2))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4, 'encoder_dim': 4}]
