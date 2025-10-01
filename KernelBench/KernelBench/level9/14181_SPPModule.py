import torch
import torch.nn as nn
import torch.nn.functional as F


class SPPModule(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPModule, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        _bs, _c, _h, _w = x.size()
        pooling_layers = [x]
        for i in range(self.num_levels):
            kernel_size = 4 * (i + 1) + 1
            padding = (kernel_size - 1) // 2
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=1,
                    padding=padding)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=1,
                    padding=padding)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_levels': 4}]
