import torch
import torch.nn as nn
from torch.nn import init


class MiniBatchDiscrimination(nn.Module):
    """
    source: https://gist.github.com/t-ae/732f78671643de97bbe2c46519972491
    paper: Salimans et al. 2016. Improved Methods for Training GANs
    """

    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features,
            kernel_dims), requires_grad=True)
        init.normal_(self.T, 0, 1)

    def forward(self, x):
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)
        M = matrices.unsqueeze(0)
        M_T = M.permute(1, 0, 2, 3)
        norm = torch.abs(M - M_T).sum(3)
        expnorm = torch.exp(-norm)
        o_b = expnorm.sum(0) - 1
        if self.mean:
            o_b /= x.size(0) - 1
        x = torch.cat([x, o_b], 1)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'kernel_dims': 4}]
