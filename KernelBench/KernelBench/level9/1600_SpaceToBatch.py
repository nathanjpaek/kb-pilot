import torch
import torch.nn as nn
import torch.utils.data


class SpaceToDim(nn.Module):

    def __init__(self, scale_factor, dims=(-2, -1), dim=0):
        super(SpaceToDim, self).__init__()
        self.scale_factor = scale_factor
        self.dims = dims
        self.dim = dim

    def forward(self, x):
        _shape = list(x.shape)
        shape = _shape.copy()
        dims = [x.dim() + self.dims[0] if self.dims[0] < 0 else self.dims[0
            ], x.dim() + self.dims[1] if self.dims[1] < 0 else self.dims[1]]
        dims = [max(abs(dims[0]), abs(dims[1])), min(abs(dims[0]), abs(dims
            [1]))]
        if self.dim in dims:
            raise RuntimeError("Integrate dimension can't be space dimension!")
        shape[dims[0]] //= self.scale_factor
        shape[dims[1]] //= self.scale_factor
        shape.insert(dims[0] + 1, self.scale_factor)
        shape.insert(dims[1] + 1, self.scale_factor)
        dim = self.dim if self.dim < dims[1] else self.dim + 1
        dim = dim if dim <= dims[0] else dim + 1
        x = x.reshape(*shape)
        perm = [dim, dims[1] + 1, dims[0] + 2]
        perm = [i for i in range(min(perm))] + perm
        perm.extend(i for i in range(x.dim()) if i not in perm)
        x = x.permute(*perm)
        shape = _shape
        shape[self.dim] *= self.scale_factor ** 2
        shape[self.dims[0]] //= self.scale_factor
        shape[self.dims[1]] //= self.scale_factor
        return x.reshape(*shape)

    def extra_repr(self):
        return f'scale_factor={self.scale_factor}'


class SpaceToBatch(nn.Module):

    def __init__(self, block_size):
        super(SpaceToBatch, self).__init__()
        self.body = SpaceToDim(block_size, dim=0)

    def forward(self, x):
        return self.body(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'block_size': 4}]
