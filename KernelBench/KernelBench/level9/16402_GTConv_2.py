import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


class GTConv_2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GTConv_2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, A):
        A = A.permute(2, 0, 1)
        A = torch.sum(F.softmax(self.weight, dim=0) * A, dim=0)
        return A


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
