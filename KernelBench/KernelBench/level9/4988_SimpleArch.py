import torch
import torch.nn as nn


class SimpleArch(nn.Module):

    def __init__(self, input_size, dropout=0.1, hidden_layer_size=10,
        output_neurons=1):
        """
        A simple architecture wrapper -- build with intuitive Sklearn-like API.
        """
        super(SimpleArch, self).__init__()
        self.h1 = nn.Linear(input_size, hidden_layer_size)
        self.h2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.h3 = nn.Linear(hidden_layer_size, 16)
        self.h4 = nn.Linear(16, output_neurons)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()
        self.sigma = nn.Sigmoid()

    def forward(self, x):
        """
        The standard forward pass. See the original paper for the formal description of this part of the DRMs. 
        """
        out = self.h1(x)
        out = self.drop(out)
        out = self.act(out)
        out = self.h2(out)
        out = self.drop(out)
        out = self.act(out)
        out = self.h3(out)
        out = self.drop(out)
        out = self.act(out)
        out = self.h4(out)
        out = self.sigma(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
