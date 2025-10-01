import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


class BaselineEstimator(nn.Module):

    def __init__(self, input_size):
        super(BaselineEstimator, self).__init__()
        self.ff1 = nn.Linear(input_size, input_size * 4)
        self.ff2 = nn.Linear(input_size * 4, 1)

    def forward(self, input, mean=False):
        input = input.detach()
        if mean:
            input = input.mean(axis=0)
        out = self.ff1(input)
        out = F.relu(out)
        out = self.ff2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
