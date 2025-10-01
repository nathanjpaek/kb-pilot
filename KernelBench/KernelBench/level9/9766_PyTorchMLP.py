import torch
import torch.nn as nn


class PyTorchMLP(nn.Module):
    """
    A feed forward network to make single step predictions on 1D time series data.
    """

    def __init__(self, inputsize, prefix):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=inputsize, out_features=round(
            inputsize / 2))
        self.fc2 = nn.Linear(in_features=round(inputsize / 2), out_features=1)
        self.act = nn.ReLU()
        self.prefix = prefix

    def forward(self, x):
        y = torch.squeeze(x)
        output = self.fc1(y)
        output = self.act(output)
        output = self.fc2(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inputsize': 4, 'prefix': 4}]
