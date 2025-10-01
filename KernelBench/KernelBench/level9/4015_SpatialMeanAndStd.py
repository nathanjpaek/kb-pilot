import torch
import torch.nn.functional
import torch.nn as nn
import torch.nn.init
import torch.onnx


class SpatialMeanAndStd(nn.Module):

    def __init__(self, shape, eps=0.0001, half_size=1.0):
        super(SpatialMeanAndStd, self).__init__()
        p = torch.empty((2, shape[0], shape[1]), dtype=torch.float32)
        p[0, ...] = torch.linspace(-half_size, half_size, shape[1])[None, :]
        p[1, ...] = torch.linspace(-half_size, half_size, shape[0])[:, None]
        self.position_code = nn.Parameter(p)
        self.position_code.requires_grad = False
        self._shape = shape
        self._eps = eps

    def forward(self, x):
        assert x.shape[1] == self._shape[0
            ], f'input shape {x.shape} vs expected {self._shape}'
        assert x.shape[2] == self._shape[1
            ], f'input shape {x.shape} vs expected {self._shape}'
        mean = torch.sum(x[:, None, :, :] * self.position_code[None, ...],
            dim=[2, 3])
        diff = self.position_code[None, ...] - mean[..., None, None]
        diff_squared = diff * diff
        std = torch.sqrt(torch.sum(x[:, None, :, :] * diff_squared, dim=[2,
            3]) + self._eps)
        return mean, std


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'shape': [4, 4]}]
