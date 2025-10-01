import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class FCNet(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 5)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(5, output_size)

    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        return output.view(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
