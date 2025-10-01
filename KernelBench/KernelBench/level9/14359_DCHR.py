import torch
import torch.nn as nn


class DCHR(nn.Module):

    def __init__(self, stride):
        super(DCHR, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=stride)

    def forward(self, x):
        pool = self.pool(x)
        shape = pool.shape
        shape = [i for i in shape]
        shape[1] = shape[1] // 2
        fill = x.new_zeros(shape)
        return torch.cat((fill, pool, fill), 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'stride': 1}]
