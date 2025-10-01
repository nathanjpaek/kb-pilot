import torch
import torch.nn as nn


class InnerProductProbe(nn.Module):

    def __init__(self, length: 'int', max_rank: 'int'=None):
        super().__init__()
        self.length = length
        if max_rank is None:
            max_rank = length
        self.b = nn.Parameter(torch.empty(max_rank, length, dtype=torch.
            float32).uniform_(-0.05, 0.05), requires_grad=True)

    def forward(self, x):
        seq_len = x.size(1)
        x = torch.einsum('gh,bih->big', self.b, x)
        x = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        y = x.clone().permute(0, 2, 1, 3)
        z = x - y
        return torch.einsum('bijg,bijg->bij', z, z)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'length': 4}]
