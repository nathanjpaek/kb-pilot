import torch
from torch.nn import functional as F
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


class GatedResidualNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, context_size=
        None, dropout=0):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(input_size, hidden_size)
        self.w3 = None if context_size is None else nn.Linear(context_size,
            hidden_size, bias=False)
        self.glu = GatedLinearUnit(hidden_size, output_size, dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.residual = nn.Sequential(
            ) if input_size == output_size else nn.Linear(input_size,
            output_size)

    def forward(self, a, c=None):
        if c is not None:
            n2 = F.elu(self.w2(a) + self.w3(c))
        else:
            n2 = F.elu(self.w2(a))
        n1 = self.w1(n2)
        grn = self.layer_norm(self.residual(a) + self.glu(n1))
        return grn


class PositionWiseFeedForward(nn.Module):

    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.grn = GatedResidualNetwork(input_size=input_size, hidden_size=
            input_size, output_size=output_size, dropout=dropout)
        self.gate_add_norm = GateAddNorm(input_size=input_size, output_size
            =output_size, dropout=dropout)

    def forward(self, x, skip):
        out = self.grn(x)
        out = self.gate_add_norm(out, skip)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'dropout': 0.5}]
