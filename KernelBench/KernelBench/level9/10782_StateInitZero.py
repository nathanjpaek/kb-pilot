import torch
from torch import nn
from torchvision import models as models
import torch.onnx
import torch.nn


class StateInitZero(nn.Module):

    def __init__(self, hidden_size, num_layers=1, batch_first=False):
        super(StateInitZero, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

    def forward(self, input: 'torch.Tensor'):
        h0 = input.new_zeros((self.num_layers, input.size(0 if self.
            batch_first else 1), self.hidden_size))
        c0 = input.new_zeros((self.num_layers, input.size(0 if self.
            batch_first else 1), self.hidden_size))
        return h0, c0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
