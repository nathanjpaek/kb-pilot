import torch
import torch.nn as nn
import torch.utils.data


class GAP(nn.Module):

    def __init__(self, dimension=1):
        """
        :param dimension:
        """
        super(GAP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        return self.avg_pool(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
