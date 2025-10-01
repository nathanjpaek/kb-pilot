import torch
from torch import Tensor
from torch import nn


class CommandEmbedding(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.embedding = nn.Linear(input_size, output_size // 2)
        self.encoding = nn.Parameter(torch.rand(1, 1, output_size // 2))

    def forward(self, command: 'Tensor') ->Tensor:
        return torch.cat([self.embedding(command), self.encoding.expand(
            command.size(0), command.size(1), -1)], dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
