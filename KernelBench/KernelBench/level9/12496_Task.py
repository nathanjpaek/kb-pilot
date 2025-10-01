import torch
import torch.nn
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.cuda
import torch.cuda.nccl
import torch.backends.cudnn
import torch.backends.mkl


class Task(nn.Module):

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(2, 2))

    def forward(self, x):
        return self.p + x


def get_inputs():
    return [torch.rand([4, 4, 2, 2])]


def get_init_inputs():
    return [[], {}]
