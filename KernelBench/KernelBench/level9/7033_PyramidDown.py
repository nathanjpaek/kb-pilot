import torch
import torch.nn as nn
from torch.nn import functional as F


class PyramidDown(nn.Module):

    def __init__(self) ->None:
        super(PyramidDown, self).__init__()
        self.filter = nn.Parameter(torch.tensor([[1, 4, 6, 4, 1], [4, 16, 
            24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4,
            1]], dtype=torch.float).reshape(1, 1, 5, 5) / 256,
            requires_grad=False)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        results = []
        for i in range(x.shape[1]):
            results.append(F.conv2d(x[:, i:i + 1, :, :], self.filter,
                padding=2, stride=2))
        return torch.cat(results, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
