import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim


class MyGlobalAvgPool2d(nn.Module):

    def __init__(self, keep_dim=True):
        super(MyGlobalAvgPool2d, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.mean(3, keepdim=self.keep_dim).mean(2, keepdim=self.keep_dim)

    def __repr__(self):
        return 'MyGlobalAvgPool2d(keep_dim=%s)' % self.keep_dim


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
