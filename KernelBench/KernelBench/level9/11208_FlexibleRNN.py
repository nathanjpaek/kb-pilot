import torch
import numpy as np
from torch import nn


def create_diag_(A, diag):
    """ This code comes is extracted from https://github.com/Lezcano/expRNN, we just repeat it as it is needed by our experiment"""
    n = A.size(0)
    diag_z = torch.zeros(n - 1)
    diag_z[::2] = diag
    A_init = torch.diag(diag_z, diagonal=1)
    A_init = A_init - A_init.T
    with torch.no_grad():
        A.copy_(A_init)
        return A


def henaff_init_(A):
    """ This code comes is extracted from https://github.com/Lezcano/expRNN, we just repeat it as it is needed by our experiment"""
    size = A.size(0) // 2
    diag = A.new(size).uniform_(-np.pi, np.pi)
    A_init = create_diag_(A, diag)
    I = torch.eye(A_init.size(0))
    return torch.mm(torch.inverse(I + A_init), I - A_init)


class modrelu(nn.Module):
    """ This code comes is extracted from https://github.com/Lezcano/expRNN, we just repeat it as it is needed by our experiment"""

    def __init__(self, features):
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)
        return phase * magnitude


class FlexibleRNN(nn.Module):
    """ This code comes is extracted (and slightly modified) from https://github.com/Lezcano/expRNN, we just repeat it as it is needed by our experiment"""

    def __init__(self, input_size, hidden_size):
        super(FlexibleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_kernel = nn.Linear(in_features=self.hidden_size,
            out_features=self.hidden_size, bias=False)
        self.input_kernel = nn.Linear(in_features=self.input_size,
            out_features=self.hidden_size, bias=False)
        self.nonlinearity = modrelu(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity
            ='relu')
        nn.init.zeros_(self.recurrent_kernel.weight.data)
        self.recurrent_kernel.weight.data = henaff_init_(self.
            recurrent_kernel.weight.data)

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        out = self.nonlinearity(out)
        return out, out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
