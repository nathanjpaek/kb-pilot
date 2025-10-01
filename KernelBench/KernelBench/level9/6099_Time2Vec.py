import torch
from torch import nn


class Time2Vec(nn.Module):
    """Encode time information

    phi and omega has k + 1 elements per each time step
    so, from input (batch_size, sample_size) will be
    ouptut (batch_size, sample_size, embed_size)

    Reference
    * https://arxiv.org/abs/1907.05321
    * https://github.com/ojus1/Time2Vec-PyTorch
    """

    def __init__(self, input_size, embed_size):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.lin = nn.Linear(self.input_size, 1)
        self.nonlin = nn.Linear(self.input_size, self.embed_size - 1)
        self.F = torch.sin

    def forward(self, x):
        """Compute following equation

        t2v(t)[i] = omega[i] * x[t] + phi[i] if i == 0
        t2v(t)[i] = f(omega[i] * x[t] + phi[i]) if 1 <= i <= k

        so, just applying Linear layer twice

        x: (batch_size, feature_size, sample_size)
        v1: (batch_size, feature_size, 1)
        v2: (batch_size, feature_size, embed_size-1)
        """
        _ = x.size(0)
        v1 = self.lin(x)
        v2 = self.F(self.nonlin(x))
        return torch.cat([v1, v2], dim=2)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'embed_size': 4}]
