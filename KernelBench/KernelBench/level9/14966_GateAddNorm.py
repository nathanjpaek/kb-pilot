import torch
from torch import nn
from torchvision import models as models
import torch.onnx
import torch.nn


class GatedLinearUnit(nn.Module):

    def __init__(self, input_size, output_size, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.w4 = nn.Linear(input_size, output_size)
        self.w5 = nn.Linear(input_size, output_size)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.act(self.w4(x)) * self.w5(x)
        return x


class GateAddNorm(nn.Module):

    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.glu = GatedLinearUnit(input_size, output_size, dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x, skip):
        return self.norm(self.glu(x) + skip)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'dropout': 0.5}]
