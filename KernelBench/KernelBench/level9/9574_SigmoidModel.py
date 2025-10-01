import torch
import torch.nn as nn


class SigmoidModel(nn.Module):
    """
        Model architecture from:
        https://medium.com/coinmonks/create-a-neural-network-in
            -pytorch-and-make-your-life-simpler-ec5367895199
    """

    def __init__(self, num_in, num_hidden, num_out):
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.lin1 = nn.Linear(num_in, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_out)
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        lin1 = self.lin1(input)
        lin2 = self.lin2(self.relu1(lin1))
        return self.sigmoid(lin2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_in': 4, 'num_hidden': 4, 'num_out': 4}]
