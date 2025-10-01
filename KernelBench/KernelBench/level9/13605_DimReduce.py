import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import torch.distributed


def GLU(input):
    out_dim = input.shape[2] // 2
    a, b = torch.split(input, out_dim, dim=2)
    return a * F.sigmoid(b)


class DimReduce(nn.Module):

    def __init__(self, input_dim, out_dim, kernel_size):
        super().__init__()
        self.convout = nn.Conv1d(input_dim, out_dim * 2, kernel_size,
            padding=kernel_size // 2)
        nn.init.xavier_normal_(self.convout.weight)

    def forward(self, input):
        input = input.transpose(1, 2)
        input = self.convout(input)
        input = input.transpose(1, 2)
        out = GLU(input)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'out_dim': 4, 'kernel_size': 4}]
