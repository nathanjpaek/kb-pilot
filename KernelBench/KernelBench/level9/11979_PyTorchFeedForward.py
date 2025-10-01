import torch
import torch.nn
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.cuda


class PyTorchFeedForward(nn.Module):

    def __init__(self, depth, width, input_size, output_size):
        super(PyTorchFeedForward, self).__init__()
        self.linears = [nn.Linear(input_size, width)]
        for i in range(depth - 1):
            self.linears.append(nn.Linear(width, width))
        self.linears.append(nn.Linear(width, output_size))
        for i, child in enumerate(self.linears):
            self.add_module('child%d' % i, child)

    def forward(self, x):
        y = F.dropout(F.relu(self.linears[0](x)), self.training)
        for layer in self.linears[1:-1]:
            y = F.relu(layer(y))
            y = F.dropout(y, self.training)
        y = F.log_softmax(self.linears[-1](y))
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'depth': 1, 'width': 4, 'input_size': 4, 'output_size': 4}]
