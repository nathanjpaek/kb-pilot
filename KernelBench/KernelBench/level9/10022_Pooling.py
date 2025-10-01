import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class Pooling(nn.Module):

    def __init__(self, pooling_type=['GAP']):
        super(Pooling, self).__init__()
        self.pooling = []
        for method in pooling_type:
            if method == 'GAP':
                self.pooling.append(F.avg_pool2d)
            elif method == 'GMP':
                self.pooling.append(F.max_pool2d)

    def forward(self, input_tensor):
        adaptiveAvgPoolWidth = input_tensor.shape[2]
        x_list = []
        for pooling in self.pooling:
            x = pooling(input_tensor, kernel_size=adaptiveAvgPoolWidth)
            x = x.view(x.size(0), -1)
            x_list.append(x)
        x = sum(x_list)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
