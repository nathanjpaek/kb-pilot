import torch
from sklearn.metrics.pairwise import *
from torch.optim.lr_scheduler import *


class GCNConv_diag(torch.nn.Module):
    """
    A GCN convolution layer of diagonal matrix multiplication
    """

    def __init__(self, input_size, device):
        super(GCNConv_diag, self).__init__()
        self.W = torch.nn.Parameter(torch.ones(input_size))
        self.input_size = input_size

    def init_para(self):
        self.W = torch.nn.Parameter(torch.ones(self.input_size))

    def forward(self, input, A, sparse=False):
        hidden = input @ torch.diag(self.W)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'device': 0}]
