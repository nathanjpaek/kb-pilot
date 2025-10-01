import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import scipy.sparse as sparse
from torch.nn.modules.utils import _pair


class SparseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, k, rho_init,
        rho_maximum, mu, stride=1, padding=0, bias=True):
        super(SparseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
            *self.kernel_size))
        self.v = nn.Parameter(torch.Tensor(out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.k = k
        self.rho_init = rho_init
        self.rho = rho_init
        self.rho_maximum = rho_maximum
        self.mu = mu
        self.y1 = np.zeros([out_channels, 1])
        self.y2 = np.zeros([out_channels, 1])
        self.z1 = np.zeros([out_channels, 1])
        self.z2 = np.zeros([out_channels, 1])
        self.v_np = np.zeros([out_channels, 1])
        self.P = sparse.csc_matrix(np.eye(self.out_channels))
        self.q = np.zeros([self.out_channels, 1])
        self.E = sparse.csc_matrix(np.vstack([np.eye(self.out_channels), np
            .ones([self.out_channels, 1]).transpose()]))
        self.l = np.vstack([np.zeros([self.out_channels, 1]), self.k * np.
            ones([1, 1])])
        self.u = np.vstack([np.ones([self.out_channels, 1]), self.k * np.
            ones([1, 1])])

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / sum(self.weight.size()))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
        v_shape = self.v.data.numpy().shape
        np_v = np.ones(v_shape)
        self.v.data = torch.from_numpy(np_v).float()

    def forward(self, input):
        return F.conv2d(input, torch.diag(self.v).mm(self.weight.view(self.
            out_channels, self.in_channels * self.kernel_size[0] * self.
            kernel_size[1])).view_as(self.weight), self.bias, self.stride,
            self.padding)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'k':
        4, 'rho_init': 4, 'rho_maximum': 4, 'mu': 4}]
