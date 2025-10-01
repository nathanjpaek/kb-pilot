import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class MaxMin(nn.Module):

    def __init__(self):
        super(MaxMin, self).__init__()

    def forward(self, x):
        y = torch.reshape(x, (x.shape[0], x.shape[1] // 2, 2, x.shape[2], x
            .shape[3]))
        maxes, _ = torch.max(y, 2)
        mins, _ = torch.min(y, 2)
        maxmin = torch.cat((maxes, mins), dim=1)
        return maxmin

    def extra_repr(self):
        return 'num_units: {}'.format(self.num_units)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
