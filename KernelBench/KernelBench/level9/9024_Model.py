import torch
from torch import Tensor
from torch.functional import Tensor
from torch import Tensor
from torch import nn


class Model(nn.Module):

    def __init__(self, input_n: 'int', output_n: 'int', hidden_n: 'int'
        ) ->None:
        super().__init__()
        self.input_shape = input_n,
        self.output_shape = output_n,
        self.hidden_n = hidden_n
        self.acctivate = nn.Softplus()
        self.fc1 = nn.Linear(input_n, self.hidden_n)
        self.fc2 = nn.Linear(self.hidden_n, self.hidden_n)
        self.fc3 = nn.Linear(self.hidden_n, output_n)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.fc1(x)
        x = self.acctivate(x)
        x = self.fc2(x)
        x = self.acctivate(x)
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_n': 4, 'output_n': 4, 'hidden_n': 4}]
