import torch
import torch.nn as nn


class AutoregressiveShift(nn.Module):
    """Shifts input right to make model autoregressive."""

    def __init__(self, embed_dim):
        super(AutoregressiveShift, self).__init__()
        self.embed_dim = embed_dim
        self.first_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.first_token)

    def forward(self, x):
        first_token = self.first_token.expand(1, x.shape[1], self.embed_dim)
        return torch.cat([first_token, x[:-1]], dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4}]
