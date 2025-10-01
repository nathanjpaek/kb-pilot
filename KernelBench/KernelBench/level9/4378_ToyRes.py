import torch
import torch.nn as nn
import torch.multiprocessing


class ToyResLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self):
        super().__init__()
        aprime = torch.Tensor(1)
        bprime = torch.Tensor(1)
        self.aprime = nn.Parameter(aprime)
        self.bprime = nn.Parameter(bprime)
        nn.init.uniform_(self.aprime)
        nn.init.uniform_(self.bprime)

    def forward(self, x):
        w = self.aprime ** 3 * (self.aprime - 3 * self.bprime + 27 * self.
            bprime ** 3)
        return x * w


class ToyRes(nn.Module):

    def __init__(self):
        super().__init__()
        self.ToyResLayer = ToyResLayer()

    def forward(self, x):
        return self.ToyResLayer(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
