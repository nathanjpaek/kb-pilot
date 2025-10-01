import math
import torch


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding for Transformer

    Parameters
    ----------
    hidden_size : `int`, required
        Hidden size of positional encoding.
        Must match hidden size of input tokens.
    dropout : `float`, required
        Dropout probability after positional encoding addition.
        If None dropout is not considered.
    max_len : `int`, optional (default = `5000`)
        Maximum sequence length to construct Positional Encoding.
    """

    def __init__(self, hidden_size: 'int', dropout: 'float'=0.0, max_len:
        'int'=5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.
            float) * -(math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self._pos_enc = torch.nn.Parameter(pe.unsqueeze(0), requires_grad=False
            )
        if dropout > 0:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def forward(self, tokens: 'torch.Tensor') ->torch.Tensor:
        tokens = tokens + self._pos_enc[:, :tokens.size(1)]
        return self._dropout(tokens)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
