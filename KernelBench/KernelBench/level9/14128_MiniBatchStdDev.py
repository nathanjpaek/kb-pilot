import torch
import torch.nn as nn
import torch.utils.cpp_extension


class MiniBatchStdDev(nn.Module):
    """Mini-Batch Standard Deviation"""

    def __init__(self, group_size: 'int'=4, eps: 'float'=0.0001) ->None:
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        B, C, H, W = x.size()
        y = x
        groups = self._check_group_size(B)
        y = y.view(groups, -1, C, H, W)
        y = y - y.mean(0, keepdim=True)
        y = y.square().mean(0)
        y = y.add_(self.eps).sqrt()
        y = y.mean([1, 2, 3], keepdim=True)
        y = y.repeat(groups, 1, H, W)
        return torch.cat([x, y], dim=1)

    def _check_group_size(self, batch_size: 'int') ->int:
        if batch_size % self.group_size == 0:
            return self.group_size
        else:
            return batch_size


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
